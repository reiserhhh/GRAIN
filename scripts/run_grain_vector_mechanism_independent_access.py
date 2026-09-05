#!/usr/bin/env python3
"""Run the frozen independent-author historical-access confirmation once."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.independent_access import (
    ACCESS_SYSTEM,
    answer_code_controls,
    confirmation_bundle,
    confirmation_conditions,
)
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from grain_research.vector_mechanisms.natural_patch import match_quantized_row_norms
from grain_research.vector_mechanisms.sensitivity import (
    local_margin_gradient,
    local_prediction,
    unit_matrix_direction,
)
from grain_research.vector_mechanisms.trace import DecoderObserver, Patch
from scripts.run_grain_vector_mechanism_controls import write_json

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SITE = "block_10"


def make_arms(h, g, budget, key, quantize):
    """Construct one target and two isotropic controls at each signed budget."""
    direction = unit_matrix_direction(g)
    if not np.any(direction):
        raise RuntimeError("Zero source derivative: retain as failed instrument")
    rows = []
    for sign in (1, -1):
        value = quantize(h + sign * budget * direction)
        rows.append(("candidate", sign, value, 0.0))
        for ri in range(2):
            seed = int(material_hash([key, ri, "independent_access_random"])[:16], 16)
            random = sign * np.random.default_rng(seed).normal(size=h.shape)
            matched, _, error = match_quantized_row_norms(
                h.reshape(1, -1),
                random.reshape(1, -1),
                value.reshape(1, -1),
                quantize,
            )
            if error.max() > 0.01:
                raise RuntimeError("Postcast random norm mismatch exceeds 1%")
            rows.append(
                (
                    f"random_{ri}",
                    sign,
                    quantize(matched.reshape(h.shape)),
                    float(error.max()),
                )
            )
    return rows


def main():
    """Open fixed holdout materials only after committing the bounded protocol."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--discovery-views", type=Path, required=True)
    p.add_argument("--calibration-views", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Never overwrite a confirmation run")
    archive_source(ROOT, out / "source_snapshot.zip")
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    views = read(a.views)
    ids = [b["author_id"] for b in views]
    previous = read(a.discovery_views) + read(a.calibration_views)
    if (
        len(views) != 48
        or len(set(ids)) != 48
        or set(ids) & {b["author_id"] for b in previous}
    ):
        raise ValueError("Expected 48 independent mechanism-holdout authors")
    bundles = [confirmation_bundle(b) for b in views]
    write_json(
        out / "access_record.json",
        {
            "protocol_commit": "4d69362",
            "source_hash": material_hash(views),
            "authors_opened": 48,
            "psychological_labels": False,
            "historical_corpus_exposure": True,
        },
    )
    torch.manual_seed(731)
    torch.set_num_threads(4)
    observer = DecoderObserver.load(
        MODEL,
        revision=REVISION,
        device="cuda",
        dtype="bfloat16",
        max_tokens=1552,
        pad_to_length=1536,
        local_files_only=True,
    )
    candidates = [
        observer.tokenizer.encode(x, add_special_tokens=False) for x in ("Yes", "No")
    ]
    if any(len(c) != 1 for c in candidates):
        raise ValueError("Single-token answer code required")
    yes, no = candidates[0][0], candidates[1][0]
    jobs, eligibility = [], []
    for bi, bundle in enumerate(bundles):
        conditions = confirmation_conditions(bundle)
        lengths = []
        for condition in conditions:
            condition["ids"] = list(
                observer.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": ACCESS_SYSTEM},
                        {"role": "user", "content": condition["user"]},
                    ],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
            lengths.append(len(condition["ids"]))
        eligible = max(lengths) <= 1536
        eligibility.append(
            {
                "author_id": bundle["author_id"],
                "eligible": eligible,
                "max_tokens": max(lengths),
                "reason": "OK" if eligible else "TOKEN_BUDGET",
            }
        )
        if not eligible:
            continue
        for condition in conditions:
            if (
                observer.prompt_ids(ACCESS_SYSTEM, condition["user"])
                != condition["ids"]
            ):
                raise ValueError("Preflight and execution tokenization differ")
            for region in ("history", "current"):
                condition[region + "_positions"] = observer.content_intervals(
                    ACCESS_SYSTEM,
                    condition["user"],
                    [condition[region + "_interval"]],
                )
            condition.pop("user")
        first = conditions[0]
        stop = max(first["history_positions"]) + 1
        for c in conditions:
            if (
                c["history_positions"] != first["history_positions"]
                or c["ids"][:stop] != first["ids"][:stop]
            ):
                raise ValueError("Historical token prefix changed")
        jobs.append(
            {
                "author_id": bundle["author_id"],
                "source_index": bi,
                "conditions": conditions,
            }
        )
    pd.DataFrame(eligibility).to_csv(out / "eligibility.csv", index=False)
    manifest = {
        "protocol": "GRAIN_VM04_INDEPENDENT_ACCESS_V1",
        "protocol_commit": "4d69362",
        "source_view_sha256": material_hash(views),
        "code_sha256": code_fingerprint(ROOT),
        "observer": observer.manifest(),
        "system": ACCESS_SYSTEM,
        "site": SITE,
        "epsilon": 0.002,
        "max_calls": 2048,
        "max_seconds": 1800,
        "jobs": jobs,
        "authors_opened": 48,
        "eligible_authors": len(jobs),
        "labels_used": False,
        "model_weight_training": False,
        "random_matching": "total Frobenius norm after BF16, not token-wise",
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    if len(jobs) < 24:
        write_json(
            out / "run_status.json",
            {"status": "INSUFFICIENT_COVERAGE", "authors": len(jobs)},
        )
        return
    quantize = lambda x: (
        torch.as_tensor(x).to(torch.bfloat16).float().numpy().astype(float)
    )
    cache = out / "private_cache"
    cache.mkdir()
    rows, checks, codes = [], [], []
    start = time.monotonic()
    stage = "RUNNING"
    write_json(out / "run_status.json", {"status": stage})
    try:
        controlled = answer_code_controls()
        for c in controlled:
            tokens = observer.prompt_ids(ACCESS_SYSTEM, c["user"])
            _, logits = observer.capture(tokens)
            margin = float(logits[yes] - logits[no])
            codes.append(
                {k: v for k, v in c.items() if k != "user"}
                | {
                    "margin": margin,
                    "correct": margin != 0 and (margin > 0) == c["expected_yes"],
                }
            )
        _, replay = observer.capture(
            observer.prompt_ids(ACCESS_SYSTEM, controlled[0]["user"])
        )
        if float(replay[yes] - replay[no]) != codes[0]["margin"]:
            raise RuntimeError("Synthetic code replay failed")
        pd.DataFrame(codes).to_csv(out / "code_controls.csv", index=False)
        for ji, job in enumerate(jobs):
            if time.monotonic() - start > 1800 or observer.calls + 40 > 2048:
                stage = "PARTIAL_BUDGET"
                break
            source_h, source_g, budget, arms = None, None, None, None
            for c in job["conditions"]:
                hp, cp = c["history_positions"], c["current_positions"]
                states, base_logits = observer.capture(
                    c["ids"], positions=[hp[0], hp[-1]]
                )
                base = float(base_logits[yes] - base_logits[no])
                h, g, margin = local_margin_gradient(observer, c["ids"], SITE, yes, no)
                if margin != base or not np.array_equal(
                    h[[hp[0], hp[-1]]], states[SITE]
                ):
                    raise RuntimeError("Autograd instrument does not replay capture")
                hh, gg = h[hp].astype(float), g[hp].astype(float)
                if c["condition"] == "source":
                    source_h, source_g = hh.copy(), gg.copy()
                    budget = 0.002 * float(np.linalg.norm(h[cp].astype(float)))
                    arms = make_arms(
                        source_h, source_g, budget, job["author_id"], quantize
                    )
                    np.savez_compressed(
                        cache / f"j{ji:02d}_source_state.npz", state=source_h
                    )
                elif not np.array_equal(hh, source_h):
                    raise RuntimeError(
                        "Historical states changed across future suffixes"
                    )
                np.savez_compressed(
                    cache / f"j{ji:02d}_{c['condition']}_gradient.npz", gradient=gg
                )
                cosine = float(
                    np.sum(gg * source_g)
                    / max(np.linalg.norm(gg) * np.linalg.norm(source_g), 1e-30)
                )
                _, sham = observer.capture(c["ids"], patch=Patch(SITE, hp, hh))
                sham_error = abs(float(sham[yes] - sham[no]) - base)
                if sham_error != 0:
                    raise RuntimeError("Self replacement failed")
                for method, sign, value, match_error in arms:
                    _, logits = observer.capture(c["ids"], patch=Patch(SITE, hp, value))
                    actual = float(logits[yes] - logits[no]) - base
                    predicted = local_prediction(gg, hh, value)
                    rows.append(
                        {
                            "author_id": job["author_id"],
                            "condition": c["condition"],
                            "polarity": c["polarity"],
                            "method": method,
                            "sign": sign,
                            "baseline_margin": base,
                            "delta": actual,
                            "predicted_delta": predicted,
                            "effect_norm": float(np.linalg.norm(value - hh)),
                            "budget": budget,
                            "norm_match_error": match_error,
                            "gradient_cosine": cosine,
                        }
                    )
                _, restored = observer.capture(c["ids"])
                restore_error = abs(float(restored[yes] - restored[no]) - base)
                if restore_error != 0:
                    raise RuntimeError("Restoration failed")
                checks.append(
                    {
                        "author_id": job["author_id"],
                        "condition": c["condition"],
                        "sham_error": sham_error,
                        "restore_error": restore_error,
                        "gradient_forward_error": abs(margin - base),
                        "prefix_error": 0.0,
                    }
                )
            pd.DataFrame(rows).to_csv(out / "interventions.csv", index=False)
            pd.DataFrame(checks).to_csv(out / "checks.csv", index=False)
            write_json(
                out / "run_status.json",
                {"status": "RUNNING", "authors": ji + 1, "calls": observer.calls},
            )
            print(
                f"INDEPENDENT {ji + 1}/{len(jobs)} calls={observer.calls}", flush=True
            )
        else:
            stage = "COMPLETE"
    except Exception as exc:
        write_json(
            out / "run_status.json",
            {
                "status": "FAILED_CONTROL_OR_RUNTIME",
                "calls": observer.calls,
                "error": type(exc).__name__ + ": " + str(exc),
            },
        )
        raise
    status = {
        "status": stage,
        "authors": len(checks) // 4,
        "eligible_authors": len(jobs),
        "rows": len(rows),
        "calls": observer.calls,
        "gradient_forwards": len(checks),
        "input_tokens": observer.input_tokens,
        "processed_tokens": observer.processed_tokens,
        "seconds": time.monotonic() - start,
        "psychological_labels": False,
        "maximum_norm_error": max((r["norm_match_error"] for r in rows), default=0),
        "all_instrument_controls_exact": all(
            all(
                r[k] == 0
                for k in (
                    "sham_error",
                    "restore_error",
                    "gradient_forward_error",
                    "prefix_error",
                )
            )
            for r in checks
        ),
    }
    write_json(out / "run_status.json", status)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
