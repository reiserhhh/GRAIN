#!/usr/bin/env python3
"""Bounded VM05 vector collection, identical author/comment evidence, no labels."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.context import QUESTIONS, SYSTEM
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_material import (
    ARMS,
    masked_comment,
    pack_features,
    shard_material,
)
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.run_grain_vector_mechanism_baselines import MODELS
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_multicomment_qwen import render_evidence

QWEN_REV = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SITES = ("block_04", "block_10", "block_16", "final_norm")


def model_signature(metadata):
    """Exclude observation time but bind every score-defining observer field."""
    return material_hash({k: v for k, v in metadata.items() if k != "unix_time"})


def bundle_arms(bundle):
    """Deterministically preserve original evidence and use the frozen cue/placebo masks."""
    original = [c for v in bundle["views"] for c in v]
    arms = [masked_comment(c["body"], c["comment_id"])[0] for c in original]
    return {arm: [x[arm] for x in arms] for arm in ARMS}


def collect_encoder(texts, tokenizer, model, prefix, pool, *, device, batch_size=4):
    """Encode complete comments, fixed FP32 shape, without hidden truncation."""
    result = []
    for i in range(0, len(texts), batch_size):
        strings = [prefix + t for t in texts[i : i + batch_size]]
        if any(len(x) > 512 for x in tokenizer(strings, truncation=False)["input_ids"]):
            raise ValueError("Encoder input exceeds frozen budget")
        # Pad the last batch with fixed empty documents to preserve execution shape.
        count = len(strings)
        strings += [prefix] * (batch_size - count)
        batch = tokenizer(
            strings,
            padding="max_length",
            max_length=512,
            truncation=False,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            h = model(**batch).last_hidden_state
            mask = batch["attention_mask"][:, :, None]
            v = h[:, 0] if pool == "cls" else (h * mask).sum(1) / mask.sum(1)
            out = torch.nn.functional.normalize(v, dim=1).float().cpu().numpy()
        if not np.isfinite(out).all():
            raise ValueError("Nonfinite encoder result")
        result.append(out[:count])
    return np.concatenate(result)


def encoder_author(arms, tokenizer, model, prefix, pool, device, *, repeat=False):
    """Reuse identical comment inputs across arms; do not duplicate computation/evidence."""
    unique = list(dict.fromkeys(t for ts in arms.values() for t in ts))
    values = collect_encoder(unique, tokenizer, model, prefix, pool, device=device)
    replay = 0.0
    if repeat:
        repeated = collect_encoder(
            unique, tokenizer, model, prefix, pool, device=device
        )
        replay = float(np.max(np.abs(values - repeated)))
        if replay > 1e-6:
            raise ValueError("Encoder replay exceeds FP32 tolerance")
    lookup = {t: i for i, t in enumerate(unique)}
    features = {}
    for arm, ts in arms.items():
        h = values[[lookup[t] for t in ts]].reshape(3, 12, -1)
        features[arm + "_comment_vectors"] = h
        features[arm + "_view_mean"] = h.mean(1)
        features[arm + "_view_std"] = h.std(1)
    return features, {
        "unique_comments": len(unique),
        "logical_comments": sum(map(len, arms.values())),
        "forward_batches": int(np.ceil(len(unique) / 4)) * (2 if repeat else 1),
        "replay_max": replay,
        "token_signatures": {
            arm: material_hash(
                tokenizer([prefix + t for t in ts], truncation=False)["input_ids"]
            )
            for arm, ts in arms.items()
        },
    }


def qwen_author(arms, observer, *, repeat=False):
    """Collect pre-answer means/stds and decision states with causal-prefix checks."""
    cache, features, readouts = {}, {}, []
    replay_error, prefix_error = 0.0, 0.0
    candidates = [
        observer.tokenizer.encode(x, add_special_tokens=False) for x in ("Yes", "No")
    ]
    if any(len(ids) != 1 for ids in candidates):
        raise ValueError("Frozen Qwen binary candidate tokenization changed")
    yes, no = [ids[0] for ids in candidates]
    start_calls = observer.calls
    for arm, ts in arms.items():
        for vi in range(3):
            view = [{"body": t} for t in ts[vi * 12 : (vi + 1) * 12]]
            previous = None
            for task in QUESTIONS:
                user, _, intervals = render_evidence(view, task)
                if user not in cache:
                    ids = observer.prompt_ids(SYSTEM, user)
                    indices = observer.content_intervals(SYSTEM, user, intervals)
                    if not indices or len(set(indices)) != len(indices):
                        raise ValueError("Invalid or overlapping original token spans")
                    states, logits = observer.capture(
                        ids,
                        positions=[len(ids) - 1],
                        pools=[("mean", indices), ("std", indices)],
                    )
                    if (
                        not all(np.isfinite(x).all() for x in states.values())
                        or not np.isfinite(logits).all()
                    ):
                        raise ValueError("Nonfinite Qwen states")
                    if repeat and not cache:
                        again, again_logits = observer.capture(
                            ids,
                            positions=[len(ids) - 1],
                            pools=[("mean", indices), ("std", indices)],
                        )
                        replay_error = max(
                            float(np.max(abs(logits - again_logits))),
                            max(
                                float(np.max(abs(states[k] - again[k]))) for k in states
                            ),
                        )
                        if replay_error != 0:
                            raise ValueError("Fixed-shape Qwen replay failed")
                    cache[user] = (
                        states,
                        float(logits[yes] - logits[no]),
                        len(ids),
                        len(indices),
                        material_hash(ids),
                    )
                states, margin, length, body_tokens, token_hash = cache[user]
                if previous is not None:
                    prefix_error = max(
                        prefix_error,
                        max(
                            float(np.max(abs(states[k][1:] - previous[k][1:])))
                            for k in states
                        ),
                    )
                    if prefix_error != 0:
                        raise ValueError("Pre-question feature invariance failed")
                previous = states
                for site, h in states.items():
                    key = f"{arm}_{site}_decision_{task}"
                    features.setdefault(
                        key, np.empty((3, len(h[0])), dtype=np.float32)
                    )[vi] = h[0]
                    if task == "behavior":
                        for row, kind in ((1, "body_mean"), (2, "body_std")):
                            key = f"{arm}_{site}_{kind}"
                            features.setdefault(
                                key, np.empty((3, len(h[row])), dtype=np.float32)
                            )[vi] = h[row]
                readouts.append(
                    {
                        "arm": arm,
                        "view": vi,
                        "task": task,
                        "margin": margin,
                        "prompt_tokens": length,
                        "body_tokens": body_tokens,
                        "input_token_sha256": token_hash,
                    }
                )
    return features, {
        "forwards": observer.calls - start_calls,
        "unique_prompts": len(cache),
        "replay_max": replay_error,
        "causal_prefix_max": prefix_error,
        "readouts": readouts,
    }


def main():
    """Require a same-model technical qualification before full no-label capture."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--preflight-manifest", type=Path, required=True)
    p.add_argument("--stage", choices=["qwen", *MODELS], required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--technical", action="store_true")
    p.add_argument("--technical-parent", type=Path)
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--max-seconds", type=int, default=28800)
    p.add_argument("--max-cache-bytes", type=int, default=10_000_000_000)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    a = p.parse_args()
    if not 1 <= a.max_seconds <= 28800:
        raise ValueError("Invalid execution time budget")
    if not 0 < a.max_cache_bytes <= 10_000_000_000:
        raise ValueError("Invalid cache budget")
    source = json.loads(a.views.read_text(encoding="utf-8"))
    common = json.loads(a.preflight_manifest.read_text(encoding="utf-8"))
    allowed = set(common["author_ids"])
    if not source or any(b["author_id"] not in allowed for b in source):
        raise ValueError("Source is outside the frozen common-material author set")
    if len({b["author_id"] for b in source}) != len(source):
        raise ValueError("Repeated source author")
    if a.technical:
        if len(source) != 12 or any(
            sum(b["source_group"] == g for b in source) != 4
            for g in ("big5_only", "mbti_only", "bridge")
        ):
            raise ValueError("Expected frozen 12-author technical panel")
    elif material_hash(source) != common["material_sha256"]:
        raise ValueError("Full source differs from common material")
    if a.technical and (a.shard_count != 1 or a.shard_index != 0):
        raise ValueError("Technical panel cannot be sharded")
    if not a.technical:
        source = shard_material(source, a.shard_count, a.shard_index)
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(
            "Use a new output directory; partial runs are never overwritten"
        )
    archive_source(ROOT, out / "source_snapshot.zip")
    torch.set_num_threads(4)
    torch.manual_seed(731)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This frozen collection protocol requires the .20 CUDA observer"
        )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    free, total = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    observer = None
    if a.stage == "qwen":
        observer = DecoderObserver.load(
            "Qwen/Qwen2.5-1.5B-Instruct",
            revision=QWEN_REV,
            device="cuda",
            dtype="bfloat16",
            max_tokens=3072,
            pad_to_length=3072,
            local_files_only=True,
        )
        layers = observer.model.model.layers
        observer.sites = {f"block_{i:02d}": layers[i] for i in (4, 10, 16)}
        observer.sites["final_norm"] = observer.model.model.norm
        model_meta = observer.manifest() | {"selected_sites": SITES}
    else:
        name, rev, prefix, pool = MODELS[a.stage]
        location = str(a.model_dir) if a.model_dir else name
        kw = {} if a.model_dir else {"revision": rev}
        tokenizer = AutoTokenizer.from_pretrained(
            location, local_files_only=True, trust_remote_code=False, **kw
        )
        model = (
            AutoModel.from_pretrained(
                location,
                local_files_only=True,
                trust_remote_code=False,
                dtype=torch.float32,
                attn_implementation="eager",
                **kw,
            )
            .eval()
            .to("cuda")
        )
        local_hashes = (
            {
                f: hashlib.sha256((a.model_dir / f).read_bytes()).hexdigest()
                for f in (
                    "config.json",
                    "model.safetensors",
                    "tokenizer.json",
                    "tokenizer_config.json",
                )
            }
            if a.model_dir
            else {}
        )
        model_meta = {
            "model": name,
            "revision": rev,
            "dtype": "float32",
            "device": "cuda",
            "pool": pool,
            "prefix": prefix,
            "fixed_padding": 512,
            "batch_size": 4,
            "local_file_hashes": local_hashes,
        }
    if a.technical_parent:
        pilot = json.loads((a.technical_parent / "run_status.json").read_text())
        pm = json.loads((a.technical_parent / "manifest.json").read_text())
        if (
            pilot["status"] != "COMPLETE"
            or not pm["technical"]
            or model_signature(pm["model"]) != model_signature(model_meta)
            or pm["stage"] != a.stage
            or pm["common_parent_identity"] != common["parent_identity"]
        ):
            raise ValueError("Technical qualification does not match exact model")
    elif not a.technical:
        raise ValueError(
            "Full collection requires a completed same-model technical parent"
        )
    manifest = {
        "protocol": "GRAIN_VM05_FEATURES_V1",
        "stage": a.stage,
        "technical": a.technical,
        "model": model_meta,
        "source_material_sha256": material_hash(source),
        "full_material_sha256": common["material_sha256"],
        "shard_count": a.shard_count,
        "shard_index": a.shard_index,
        "common_parent_identity": common["parent_identity"],
        "authors": [b["author_id"] for b in source],
        "code_sha256": code_fingerprint(ROOT),
        "all_arms": ARMS,
        "max_seconds": a.max_seconds,
        "max_cache_bytes": a.max_cache_bytes,
        "gpu_free_before": free,
        "gpu_total": total,
        "labels_read": False,
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    cache_dir = out / "private_cache"
    cache_dir.mkdir()
    records, cache_bytes = [], 0
    for i, b in enumerate(source):
        arms = bundle_arms(b)
        features, metrics = (
            qwen_author(arms, observer, repeat=i == 0)
            if observer
            else encoder_author(
                arms, tokenizer, model, prefix, pool, "cuda", repeat=i == 0
            )
        )
        path = cache_dir / (b["author_id"] + ".npz")
        stored, aliases = pack_features(features)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **stored)
        metrics.update(
            author_id=b["author_id"],
            source_group=b["source_group"],
            feature_sha256=hashlib.sha256(buffer.getvalue()).hexdigest(),
            feature_aliases=aliases,
            cache_format="lossless_aliases_v1",
        )
        # Reserve generous metadata space, and refuse before writing past the shared cap.
        projected = buffer.tell() + len(json.dumps(metrics, indent=2).encode()) + 1024
        if cache_bytes + projected > a.max_cache_bytes:
            write_json(
                out / "run_status.json",
                {
                    "status": "PARTIAL_RESOURCE_BOUND",
                    "authors": i,
                    "planned": len(source),
                    "cache_bytes": cache_bytes,
                    "reason": "shared cache budget",
                    "labels_read": False,
                },
            )
            return
        path.write_bytes(buffer.getvalue())
        write_json(cache_dir / (b["author_id"] + ".json"), metrics)
        cache_bytes += (
            path.stat().st_size
            + (cache_dir / (b["author_id"] + ".json")).stat().st_size
        )
        records.append(metrics)
        elapsed = time.monotonic() - start
        peak = torch.cuda.max_memory_allocated()
        status = {
            "status": "RUNNING",
            "authors": i + 1,
            "planned": len(source),
            "seconds": elapsed,
            "gpu_peak_allocated": peak,
            "gpu_free_before": free,
            "new_forwards": observer.calls
            if observer
            else sum(r["forward_batches"] for r in records),
            "logical_comments": (i + 1) * 36,
            "cache_bytes": cache_bytes,
            "labels_read": False,
        }
        write_json(out / "run_status.json", status)
        if (i + 1) % 12 == 0 or i == 0:
            print(json.dumps(status), flush=True)
        if peak > free * 0.8 or elapsed > a.max_seconds:
            write_json(
                out / "run_status.json", status | {"status": "PARTIAL_RESOURCE_BOUND"}
            )
            return
    write_json(
        out / "run_status.json",
        status | {"status": "COMPLETE", "parent_identity": manifest["identity"]},
    )
    print(json.dumps(status | {"status": "COMPLETE"}), flush=True)


if __name__ == "__main__":
    main()
