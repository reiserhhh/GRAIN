#!/usr/bin/env python3
"""Bounded direct LLM trait qualification or supported source shards, no target values."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.direct_cache import verify_collection
from grain_research.vector_mechanisms.direct_material import (
    archive_direct,
    load_registration,
    technical_cases,
    verify_prompts,
    verify_tokenizer,
)
from grain_research.vector_mechanisms.direct_traits import candidates, collect_author
from grain_research.vector_mechanisms.external_material import shard_material
from grain_research.vector_mechanisms.history_collection import observation_signature
from grain_research.vector_mechanisms.ready_comparison import digest, verified_packet
from scripts.qualify_grain_source_history import load_observer
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_external_features import bundle_arms

CODE = (
    "scripts/collect_grain_direct_traits.py",
    "grain_research/vector_mechanisms/direct_material.py",
    "grain_research/vector_mechanisms/direct_cache.py",
    "grain_research/vector_mechanisms/direct_traits.py",
    "grain_research/vector_mechanisms/trace.py",
    "grain_research/vector_mechanisms/ready_comparison.py",
    "grain_research/vector_mechanisms/history_instrument.py",
    "scripts/qualify_grain_source_history.py",
    "grain_research/vector_mechanisms/external_material.py",
    "scripts/run_grain_vector_mechanism_external_features.py",
)


def qualified_parent(folder, material, code, cases):
    """No source shard can substitute an untested or token-only technical packet."""
    if folder is None:
        raise ValueError("Real source collection requires direct qualification")
    m, s = verified_packet(folder)
    expected = technical_cases(cases)
    verify_collection(folder, expected, material, code)
    if (
        m["protocol"] != "VM05_DIRECT_TRAIT_COLLECTION_V1"
        or m["mode"] != "qualification"
        or m["material_identity"] != material["identity"]
        or m["case_commitment"] != material_hash(expected)
        or m["authors"] != [c["author_id"] for c in expected]
        or m["code_sha256"] != code
        or s["authors"] != 6
        or s["logical_prompts"] != 234
        or not s["real_neural_collection"]
        or not s["all_qualification_rows_replayed"]
        or s["replay_max"] != 0
        or m["labels_read"]
        or m["bridge_outcomes_read"]
        or s["labels_read"]
        or s["bridge_outcomes_read"]
    ):
        raise ValueError("Incomplete or changed real direct qualification")
    model = json.loads(
        (Path(folder) / "actual_observer.json").read_text(encoding="utf-8")
    )
    if (
        observation_signature(model) != s["model_signature"]
        or model["revision"] != material["qwen_revision"]
        or model["dtype"] != "torch.bfloat16"
        or not model["device"].startswith("cuda")
        or model["training"]
        or model["attention"] != "eager"
        or model["pad_to_length"] != 3072
    ):
        raise ValueError("Qualified direct observer is not pinned CUDA")
    first_pass = 0
    for c in expected:
        path = (
            Path(folder) / "private_cache" / (material_hash(c["author_id"]) + ".json")
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            data["case_commitment"] != material_hash(c)
            or data["author_id"] != c["author_id"]
        ):
            raise ValueError("Direct technical author changed")
        first_pass += data["metrics"]["forwards"]
    if first_pass <= 0 or s["forwards"] != 2 * first_pass:
        raise ValueError("Incomplete full-row qualification replay")
    return m, s


def run(a):
    """Open only registered text material; evaluate model scores, never label tables."""
    if not 60 <= a.max_seconds <= 3600 or not 0 < a.max_cache_bytes <= 1_000_000_000:
        raise ValueError("Direct collection must be time/cache bounded")
    started = time.monotonic()
    by_author, whole, material = load_registration(a.views, a.material_dir)
    verify_tokenizer(a.tokenizer_dir, material)
    a.tokenizer = AutoTokenizer.from_pretrained(
        str(a.tokenizer_dir), local_files_only=True, trust_remote_code=False
    )
    for task, alphabet in material["alphabets"].items():
        if candidates(task, a.tokenizer) != alphabet:
            raise ValueError("Direct answer alphabet changed")
    code = {name: digest(ROOT / name) for name in CODE}
    qualification = qsummary = None
    if a.qualification:
        if a.shard_count != 128 or a.shard_index != 0:
            raise ValueError("Technical qualification is six fixed cases, not a shard")
        cases = technical_cases(whole)
    else:
        cases = shard_material(whole, a.shard_count, a.shard_index)
        if not a.preflight_only:
            qualification, qsummary = qualified_parent(
                a.qualification_dir, material, code, whole
            )
    mode = (
        "preflight"
        if a.preflight_only
        else "qualification"
        if a.qualification
        else "source_shard"
    )
    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Private direct observations must remain under results")
    out.mkdir(parents=True, exist_ok=False)
    archive_direct(ROOT, out / "source_snapshot.zip")
    fields = {
        "protocol": "VM05_DIRECT_TRAIT_COLLECTION_V1",
        "mode": mode,
        "material_identity": material["identity"],
        "qualification_identity": qualification["identity"] if qualification else None,
        "case_commitment": material_hash(cases),
        "authors": [c["author_id"] for c in cases],
        "supported_authors": len(whole),
        "shard_count": a.shard_count,
        "shard_index": a.shard_index,
        "code_sha256": code,
        "source_snapshot_sha256": digest(out / "source_snapshot.zip"),
        "max_seconds": a.max_seconds,
        "max_cache_bytes": a.max_cache_bytes,
        "labels_read": False,
        "bridge_outcomes_read": False,
    }
    fields["identity"] = material_hash(fields)
    write_json(out / "manifest.json", fields)
    cache = out / "private_cache"
    cache.mkdir()
    obs = None
    records, logical, cache_bytes, replay = [], 0, 0, 0.0
    try:
        if not a.preflight_only:
            obs = load_observer(a, material)
            model = obs.manifest()
            if qsummary and observation_signature(model) != qsummary["model_signature"]:
                raise ValueError("Direct production observer changed")
            write_json(out / "actual_observer.json", model)
        for i, c in enumerate(cases):
            if time.monotonic() - started > a.max_seconds:
                raise TimeoutError("Preserve incomplete direct shard")
            arms = bundle_arms(by_author[c["author_id"]])
            expected = verify_prompts(c, arms, a.tokenizer)
            logical += len(expected)
            if obs is None:
                records.append(
                    {
                        "case_commitment": material_hash(c),
                        "logical_prompts": len(expected),
                        "token_replay": True,
                    }
                )
                continue
            rows, meta = collect_author(
                arms,
                c["tasks"],
                obs,
                repeat=not a.qualification and i == 0,
                deadline=started + a.max_seconds,
            )
            if [{k: r[k] for k in expected[0]} for r in rows] != expected:
                raise ValueError("Observed direct grid changed")
            if a.qualification:
                again, other = collect_author(
                    arms, c["tasks"], obs, deadline=started + a.max_seconds
                )
                if rows != again or meta != other:
                    raise ValueError("Every native direct technical row must replay")
            replay = max(replay, meta["replay_max"])
            packet = {
                "author_id": c["author_id"],
                "source_group": c["source_group"],
                "case_commitment": material_hash(c),
                "metrics": meta,
                "readouts": rows,
                "view_means": {
                    f"{arm}__{t}": float(
                        np.mean(
                            [
                                r["conditional_score"]
                                for r in rows
                                if r["arm"] == arm and r["task"] == t
                            ]
                        )
                    )
                    for arm in arms
                    for t in c["tasks"]
                },
            }
            data = (
                json.dumps(packet, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
            )
            if cache_bytes + len(data) > a.max_cache_bytes:
                raise MemoryError(
                    "Direct shard cache bound; preserve completed authors"
                )
            key = material_hash(c["author_id"])
            path = cache / (key + ".json")
            path.write_bytes(data)
            cache_bytes += len(data)
            records.append(
                {"cache_key": key, "sha256": digest(path), "logical_prompts": len(rows)}
            )
            write_json(
                out / "run_status.json",
                {
                    "status": "RUNNING",
                    "authors": i + 1,
                    "planned": len(cases),
                    "seconds": time.monotonic() - started,
                    "forwards": obs.calls,
                },
            )
        write_json(out / "private_records.json", records)
        summary = {
            "status": "COMPLETE",
            "parent_identity": fields["identity"],
            "authors": len(records),
            "logical_prompts": logical,
            "real_neural_collection": obs is not None,
            "forwards": obs.calls if obs else 0,
            "replay_max": replay,
            "all_qualification_rows_replayed": bool(
                a.qualification and obs is not None
            ),
            "model_signature": observation_signature(obs.manifest()) if obs else None,
            "cache_bytes": cache_bytes,
            "seconds": time.monotonic() - started,
            "labels_read": False,
            "bridge_outcomes_read": False,
            "outputs_sha256": {
                p.relative_to(out).as_posix(): digest(p)
                for p in out.rglob("*")
                if p.is_file()
                and not p.name.startswith("._")
                and p.name
                not in ("manifest.json", "run_status.json", "source_snapshot.zip")
            },
        }
        write_json(out / "summary.json", summary)
        write_json(out / "run_status.json", summary)
        print(
            json.dumps({k: v for k, v in summary.items() if k != "outputs_sha256"}),
            flush=True,
        )
    except BaseException as e:
        write_json(
            out / "run_status.json",
            {
                "status": "FAILED_STOP",
                "authors": len(records),
                "exception_type": type(e).__name__,
                "error": str(e),
                "forwards": obs.calls if obs else 0,
            },
        )
        raise


def main():
    """Preflight is token-only; real execution waits for the original CUDA queue."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("views", "material-dir", "tokenizer-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("qualification-dir", "observer-queue-dir"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--qualification", action="store_true")
    p.add_argument("--shard-count", type=int, default=128)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--max-seconds", type=int, default=1800)
    p.add_argument("--max-cache-bytes", type=int, default=200_000_000)
    run(p.parse_args())


if __name__ == "__main__":
    main()
