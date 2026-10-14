#!/usr/bin/env python3
"""Collect actual crossed articles, then compare fixed vector observation laws."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_joint_persona_style as M
from grain_research.vector_mechanisms.persona_trace import audit_text
from scripts import run_grain_semantic_anchor_hm147 as H

DEFAULT = ROOT / "results/grain_math_response_v1/joint_persona_style_r1"
MODEL = (
    ROOT
    / "results/grain_vector_mechanisms/local_observers/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)
FROZEN = ROOT / "results/v7_operator_boundary_audit/e0_full_20260714"
REPRESENTATION = FROZEN / "artifacts/common_source_comment_representation.joblib"
BUNDLE = FROZEN / "operator_bundles/whole_factor_bundle.json"
CODE = (
    "grain_research/construction/math_joint_persona_style.py",
    "scripts/run_grain_joint_persona_style_hm150.py",
    "tests/test_grain_joint_persona_style.py",
    "docs/GRAIN_JOINT_PERSONA_STYLE_PLAN_HM150.md",
)


def status(out, stage, state, **fields):
    """Atomically update an owned runtime checkpoint, never scientific outputs."""
    path = out / f"{stage}_status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"status": state, **fields}, indent=2) + "\n")
    temporary.replace(path)


def check(out):
    """Require unchanged design, code, old constructor and representation."""
    manifest = H.load(out / "manifest.json")
    for name, digest in manifest["hashes"].items():
        if H.digest(ROOT / name) != digest:
            raise ValueError("Frozen dependency changed: " + name)
    if H.digest(out / "materials.json") != manifest["materials_sha256"]:
        raise ValueError("Frozen joint design changed")
    return manifest


def prepare(out):
    """Create a full702-cell experiment independently of all article outcomes."""
    from transformers import AutoTokenizer

    old = H.load(H.SOURCE / "manifest.json")
    H.check(H.DEFAULT)
    rows = M.joint_materials(old)
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    tokenizer.padding_side = "left"
    for row in rows:
        row["prompt_ids"] = list(
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": row["system"]},
                    {"role": "user", "content": row["user"]},
                ],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        if len(row["prompt_ids"]) > 384:
            raise ValueError("Prompt exceeds fixed support")
    files = [ROOT / f for f in CODE] + [
        H.SOURCE / "manifest.json",
        H.DEFAULT / "program.npz",
        REPRESENTATION,
        BUNDLE,
        MODEL / "config.json",
        MODEL / "tokenizer.json",
        MODEL / "tokenizer_config.json",
        ROOT / "grain_research/construction/math_semantic_anchor.py",
        ROOT / "grain_research/vector_mechanisms/trace.py",
        ROOT / "grain_research/vector_mechanisms/persona_trace.py",
        ROOT / "suica_core/v7_psychometric.py",
        ROOT / "suica_core/v7_observations.py",
        ROOT / "suica_core/v7_operator_controls.py",
        ROOT / "scripts/run_grain_semantic_anchor_hm147.py",
    ]
    out.mkdir(parents=True, exist_ok=False)
    (out / "batches").mkdir()
    (out / "observations").mkdir()
    H.write_json(out / "materials.json", rows)
    H.write_json(
        out / "manifest.json",
        {
            "packet": "HM150",
            "rows": len(rows),
            "batches": 54,
            "hashes": {str(p.relative_to(ROOT)): H.digest(p) for p in files},
            "materials_sha256": H.digest(out / "materials.json"),
            "model_weights_sha256": old["model_weights_sha256_pinned"],
            "generation": old["generation"],
            "generation_seconds_cap": 2400,
            "observation_seconds_cap": 600,
            "generation_policy": "13 complete conditions per task/seed/style;leftpad;reset declared seed perbatch;no lexical-noise equivalence claim",
            "representations": [
                "qwen_input_mean",
                "qwen_block16_mean",
                "grain4_slices",
            ],
            "scope": "Syntheticassignedpersona;not humantraits,not accuracy,not assumed style-noise",
        },
    )
    print(json.dumps({"prepared_rows": len(rows), "batches": 54}), flush=True)


def batches(rows):
    """Retain first-appearance batch order and every within-batch position."""
    groups = {}
    for row in rows:
        groups.setdefault(row["batch"], []).append(row)
    return list(groups.values())


def offline_model(manifest):
    """Load one local frozen writer/observer without network or gradients."""
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable;no remote/API fallback")
    if H.digest(MODEL / "model.safetensors") != manifest["model_weights_sha256"]:
        raise ValueError("Pinned model weights changed")
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=768, local_files_only=True
    )
    observer.model.eval().requires_grad_(False)
    observer.sites = {
        "input_embedding": observer.model.model.embed_tokens,
        "block_16": observer.model.model.layers[16],
    }
    return observer


def generate(out):
    """Generate actual joint corners in immutable, resumable finite batches."""
    import torch

    manifest = check(out)
    rows = H.load(out / "materials.json")
    groups = batches(rows)
    observer = offline_model(manifest)
    if not (out / "model_manifest.json").exists():
        H.write_json(out / "model_manifest.json", observer.manifest())
    start, eos = time.monotonic(), observer.tokenizer.eos_token_id
    done = 0
    for index, group in enumerate(groups):
        destination = out / "batches" / f"{index:03d}.json"
        if destination.exists():
            cached = H.load(destination)
            if [r["text_id"] for r in cached] != [r["text_id"] for r in group]:
                raise ValueError("Stored generation batch disagrees with design")
            done += len(group)
            continue
        if time.monotonic() - start > manifest["generation_seconds_cap"]:
            status(out, "generation", "PARTIAL_BUDGET", done=done, total=len(rows))
            return
        status(
            out,
            "generation",
            "RUNNING",
            done=done,
            current_batch=index,
            pid=os.getpid(),
        )
        width = max(len(r["prompt_ids"]) for r in group)
        input_ids = torch.tensor(
            [[eos] * (width - len(r["prompt_ids"])) + r["prompt_ids"] for r in group],
            device="mps",
        )
        mask = torch.tensor(
            [
                [0] * (width - len(r["prompt_ids"])) + [1] * len(r["prompt_ids"])
                for r in group
            ],
            device="mps",
        )
        torch.manual_seed(group[0]["seed"])
        begin = time.monotonic()
        with torch.inference_mode():
            outputs = observer.model.generate(
                input_ids=input_ids,
                attention_mask=mask,
                pad_token_id=eos,
                eos_token_id=eos,
                use_cache=True,
                **manifest["generation"],
            )
        elapsed = time.monotonic() - begin
        records = []
        for row, tokens in zip(group, outputs[:, width:].tolist(), strict=True):
            finish = "eos" if eos in tokens else "length"
            ids = tokens[: tokens.index(eos)] if eos in tokens else tokens
            text = observer.tokenizer.decode(ids, skip_special_tokens=True)
            records.append(
                {
                    "text_id": row["text_id"],
                    "new_ids": ids,
                    "text": text,
                    "finish_reason": finish,
                    "n_new": len(ids),
                    "audit": audit_text(text, row["situation"], finish),
                    "batch_seconds": elapsed,
                }
            )
        H.write_json(destination, records)
        done += len(group)
        print(
            json.dumps(
                {
                    "batch": index,
                    "done": done,
                    "total": len(rows),
                    "seconds": elapsed,
                    "new_tokens": sum(r["n_new"] for r in records),
                }
            ),
            flush=True,
        )
    seal = {
        str(p.relative_to(out)): H.digest(p)
        for p in sorted((out / "batches").glob("*.json"))
    }
    H.write_json(out / "generation_seal.json", seal)
    status(
        out,
        "generation",
        "COMPLETE",
        done=done,
        total=len(rows),
        seconds=time.monotonic() - start,
    )


def generated(out):
    """Authenticate complete generated outputs; never silently use a subset."""
    if H.load(out / "generation_status.json")["status"] != "COMPLETE":
        raise ValueError("Generation is not complete")
    for name, digest in H.load(out / "generation_seal.json").items():
        if H.digest(out / name) != digest:
            raise ValueError("Generated output changed")
    rows = H.load(out / "materials.json")
    records = [
        r for path in sorted((out / "batches").glob("*.json")) for r in H.load(path)
    ]
    if [r["text_id"] for r in records] != [r["text_id"] for r in rows]:
        raise ValueError("Complete design order required")
    return rows, records


def grain_observations(records):
    """Score raw lossless slices with the existing TF-IDF/SVD factor bundle."""
    import joblib
    import pandas as pd

    from scripts.run_grain_within_comment_channels import lexical_slices
    from suica_core.v7_observations import ObservationSpec, canonicalize_comments
    from suica_core.v7_operator_controls import score_operator_from_comments
    from suica_core.v7_psychometric import read_factor_bundle

    representation, bundle = joblib.load(REPRESENTATION), read_factor_bundle(BUNDLE)
    values = np.full((len(records), 4, 4), np.nan)
    raw = []
    for i, record in enumerate(records):
        for j, text in enumerate(lexical_slices(record["text"])):
            raw.append({"author": f"{i:04d}|{j}", "body": text})
    if not raw:
        return values, 0.0
    frame = canonicalize_comments(
        pd.DataFrame(raw), order_col=None, condition_col=None, min_tokens=12
    )
    frame["split"] = "confirmation"
    _, features, scored = score_operator_from_comments(
        frame,
        spec=ObservationSpec(**bundle.operator),
        representation=representation,
        bundle=bundle,
    )
    columns = [f"SU7-FC-{i:04d}@v1_z" for i in range(1, 5)]
    selected = scored.loc[scored.score_status.eq("SCORED")].set_index("user_id")
    x = (
        features.set_index("user_id")
        .loc[selected.index, bundle.feature_names]
        .to_numpy(float)
    )
    x = (x - bundle.feature_center) / bundle.feature_scale - bundle.fa_mean
    loading = np.asarray(bundle.factor_loadings)
    weighted = loading / np.asarray(bundle.noise_variance)
    posterior = np.linalg.solve(np.eye(4) + weighted @ loading.T, weighted @ x.T).T
    replay = (posterior * bundle.factor_signs - bundle.norm_mean) / bundle.norm_scale
    result = selected[columns].to_numpy(float)
    np.testing.assert_allclose(result, replay, atol=1e-8, rtol=1e-8)
    for key, value in zip(selected.index, result, strict=True):
        i, j = map(int, key.split("|"))
        values[i, j] = value
    return values, float(abs(result - replay).max())


def observe(out):
    """Read only generated text tokens, never the writer's system/task prompts."""
    manifest = check(out)
    rows, records = generated(out)
    observer = offline_model(manifest)
    start = time.monotonic()
    for index, group in enumerate(batches(rows)):
        path = out / "observations" / f"{index:03d}.npz"
        if path.exists():
            continue
        if time.monotonic() - start > manifest["observation_seconds_cap"]:
            status(out, "observation", "PARTIAL_BUDGET", current_batch=index)
            return
        begin, a, b = time.monotonic(), [], []
        for record in records[index * 13 : (index + 1) * 13]:
            ids = observer.tokenizer.encode(record["text"], add_special_tokens=False)
            if not ids:
                a.append(np.full(1536, np.nan))
                b.append(np.full(1536, np.nan))
                continue
            if len(ids) > 768:
                raise ValueError("Text-only observation exceeds fixed support")
            captured, _ = observer.capture(ids, pools=[("mean", list(range(len(ids))))])
            a.append(captured["input_embedding"][1])
            b.append(captured["block_16"][1])
        H.write_npz(
            path, qwen_input_mean=np.asarray(a), qwen_block16_mean=np.asarray(b)
        )
        print(
            json.dumps(
                {
                    "observed_batch": index,
                    "rows": len(group),
                    "seconds": time.monotonic() - begin,
                }
            ),
            flush=True,
        )
        status(out, "observation", "RUNNING", current_batch=index, pid=os.getpid())
    arrays = {k: [] for k in manifest["representations"][:2]}
    for path in sorted((out / "observations").glob("*.npz")):
        cached = H.archive(path, arrays)
        for key, values in arrays.items():
            values.append(cached[key])
    grain, error = grain_observations(records)
    H.write_npz(
        out / "numeric_observations.npz",
        **{k: np.concatenate(v) for k, v in arrays.items()},
        grain4_slices=grain,
    )
    H.write_json(
        out / "observation_seal.json",
        {"numeric_observations.npz": H.digest(out / "numeric_observations.npz")},
    )
    status(
        out,
        "observation",
        "COMPLETE",
        rows=len(rows),
        local_forwards=observer.calls,
        grain_supported=int(np.isfinite(grain).all(axis=(1, 2)).sum()),
        fa_replay_error=error,
        seconds=time.monotonic() - start,
    )


def analyze(out):
    """Preserve all conditional interactions and both nonlinear-link allocations."""
    manifest = check(out)
    rows, records = generated(out)
    for name, digest in H.load(out / "observation_seal.json").items():
        if H.digest(out / name) != digest:
            raise ValueError("Cold numeric observation changed")
    arrays = H.archive(out / "numeric_observations.npz", manifest["representations"])
    corners, meta = M.interaction_design(rows)
    program = H.archive(
        H.DEFAULT / "program.npz", ("origin", "scale", "coefficients", "norms")
    )
    original = arrays["grain4_slices"]
    z = (original - program["origin"]) / program["scale"][None, :, None]
    arrays["grain4_slices"] = z.reshape(len(z), -1)
    contrasts, supports, coverage, stored = [], [], [], {}
    for name, values in arrays.items():
        valid = np.isfinite(values[corners]).all(axis=(1, 2))
        mixed = np.full((len(corners), values.shape[1]), np.nan)
        compact = values[corners[valid]].reshape(-1, values.shape[1])
        mixed[valid] = M.mixed_difference(
            compact, np.arange(len(compact)).reshape(-1, 4)
        )
        stored[name] = mixed
        for i, item in enumerate(meta):
            coverage.append(
                {**item, "representation": name, "supported": bool(valid[i])}
            )
            if not valid[i]:
                continue
            hs, h0, ls, l0 = values[corners[i]]
            contrasts.append(
                {
                    **item,
                    "representation": name,
                    "persona_neutral_rms": float(np.sqrt(np.mean((h0 - l0) ** 2))),
                    "persona_style_rms": float(np.sqrt(np.mean((hs - ls) ** 2))),
                    "mixed_rms": float(np.sqrt(np.mean(mixed[i] ** 2))),
                    "mixed_words": records[corners[i, 0]]["audit"]["words"]
                    - records[corners[i, 1]]["audit"]["words"]
                    - records[corners[i, 2]]["audit"]["words"]
                    + records[corners[i, 3]]["audit"]["words"],
                }
            )
        groups = {}
        for i, item in enumerate(meta):
            key = tuple(item[k] for k in ("axis", "family", "situation", "style"))
            groups.setdefault(key, []).append(i)
        for ids in groups.values():
            item = {k: v for k, v in meta[ids[0]].items() if k != "seed"}
            ok = bool(valid[ids].all())
            supports.append(
                {
                    **item,
                    "representation": name,
                    "supported": ok,
                    **(
                        M.seed_support(mixed[ids])
                        if ok
                        else {
                            k: None
                            for k in (
                                "cross_seed_support",
                                "mean_squared_norm",
                                "within_seed_term",
                                "identity_error",
                                "mean_rms",
                                "observed_rms",
                            )
                        }
                    ),
                }
            )
    valid = np.isfinite(z[corners]).all(axis=(1, 2, 3))
    compact = z[corners[valid]].reshape(-1, 4, 4)
    accounts = M.tensor_accounts(compact, np.arange(len(compact)).reshape(-1, 4))
    table = []
    ids = np.flatnonzero(valid)
    for part, values in accounts.items():
        stored["tensor_" + part] = values
    for index, row_id in enumerate(ids):
        for arm, arm_name in enumerate(H.ARMS):
            for target, target_name in enumerate(H.TARGETS):
                value = {
                    k: float(v[index] @ program["coefficients"][arm, target])
                    for k, v in accounts.items()
                }
                table.append(
                    {
                        **meta[row_id],
                        "arm": arm_name,
                        "read_target": target_name,
                        **value,
                    }
                )
    H.write_npz(out / "interactions.npz", **stored, grain_supported_corners=ids)
    H.write_csv(out / "vector_contrasts.csv", contrasts)
    H.write_csv(out / "seed_support.csv", supports)
    H.write_csv(out / "coverage.csv", coverage)
    H.write_csv(out / "score_accounts.csv", table)
    audit = [
        {
            **{
                k: r[k]
                for k in (
                    "text_id",
                    "axis",
                    "pole",
                    "family",
                    "style",
                    "situation",
                    "seed",
                    "role",
                )
            },
            **g["audit"],
            "new_tokens": g["n_new"],
        }
        for r, g in zip(rows, records, strict=True)
    ]
    H.write_csv(out / "text_audit.csv", audit)
    summary = {
        "rows": len(rows),
        "interactions": len(meta),
        "score_rows": len(table),
        "audit": {
            "refusals": sum(g["audit"]["refusal"] for g in records),
            "explicit_trait_texts": sum(g["audit"]["leak_hits"] > 0 for g in records),
            "truncated": sum(g["audit"]["truncated"] for g in records),
        },
        "representations": {
            name: {
                "supported_groups": sum(
                    s["supported"] and s["representation"] == name for s in supports
                ),
                "positive_cross_seed_groups": sum(
                    s["supported"]
                    and s["representation"] == name
                    and s["cross_seed_support"] > 0
                    for s in supports
                ),
                "mean_cross_seed_support": float(
                    np.mean(
                        [
                            s["cross_seed_support"]
                            for s in supports
                            if s["supported"] and s["representation"] == name
                        ]
                    )
                ),
            }
            for name in arrays
        },
        "tensor_upper_identity_error": float(
            abs(
                accounts["observed"] - accounts["upper_link"] - accounts["upper_action"]
            ).max()
        ),
        "tensor_lower_identity_error": float(
            abs(
                accounts["observed"] - accounts["lower_link"] - accounts["lower_action"]
            ).max()
        ),
        "scope": "Actualjointsyntheticoutputs;raw representations separated;no human r/AUC or denoising accuracy",
    }
    H.write_json(out / "summary.json", summary)
    files = [
        "interactions.npz",
        "vector_contrasts.csv",
        "seed_support.csv",
        "coverage.csv",
        "score_accounts.csv",
        "text_audit.csv",
        "summary.json",
    ]
    H.write_json(
        out / "output_seal.json", {name: H.digest(out / name) for name in files}
    )
    print(json.dumps(summary), flush=True)


def verify(out):
    """Read back actual corners and all frozen identities without model calls."""
    manifest = check(out)
    rows, _ = generated(out)
    for seal in ("observation_seal.json", "output_seal.json"):
        for name, digest in H.load(out / seal).items():
            if H.digest(out / name) != digest:
                raise ValueError("Scientific output changed: " + name)
    arrays = H.archive(out / "numeric_observations.npz", manifest["representations"])
    corners, _ = M.interaction_design(rows)
    program = H.archive(
        H.DEFAULT / "program.npz", ("origin", "scale", "coefficients", "norms")
    )
    z = (arrays["grain4_slices"] - program["origin"]) / program["scale"][None, :, None]
    arrays["grain4_slices"] = z.reshape(len(z), -1)
    with np.load(out / "interactions.npz", allow_pickle=False) as stored:
        error = 0.0
        for name, values in arrays.items():
            valid = np.isfinite(values[corners]).all(axis=(1, 2))
            expected = (
                values[corners[valid, 0]]
                + values[corners[valid, 3]]
                - values[corners[valid, 1]]
                - values[corners[valid, 2]]
            )
            actual = stored[name][valid]
            error = max(error, float(abs(actual - expected).max()))
            np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
        for anchor in ("upper", "lower"):
            np.testing.assert_allclose(
                stored["tensor_observed"],
                stored[f"tensor_{anchor}_link"] + stored[f"tensor_{anchor}_action"],
                atol=1e-10,
                rtol=1e-10,
            )
    print(
        json.dumps(
            {
                "verified": True,
                "alternate_raw_corner_error": error,
                "new_model_calls": 0,
            }
        ),
        flush=True,
    )


def main():
    """Run exactly one declared stage; errors remain inspectable and terminal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "generate", "observe", "analyze", "verify")
    )
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    try:
        globals()[args.stage](args.output)
    except Exception as error:
        if args.output.exists():
            status(
                args.output,
                {"generate": "generation", "observe": "observation"}.get(
                    args.stage, args.stage
                ),
                "FAILED",
                error=f"{type(error).__name__}: {error}",
                pid=os.getpid(),
            )
        raise


if __name__ == "__main__":
    main()
