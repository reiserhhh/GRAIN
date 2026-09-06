#!/usr/bin/env python3
"""Bounded full-registry history shards, with exact six-author qualification replay."""

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
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_material import (
    pack_features,
    shard_material,
    unpack_features,
)
from grain_research.vector_mechanisms.history_collection import (
    KINDS,
    SITES,
    collect_history_author,
    history_channel,
    observation_signature,
    qualification_packet,
)
from grain_research.vector_mechanisms.history_instrument import (
    archive_instrument,
    material_layouts,
    registered_cases,
    tokenizer_matches,
)
from grain_research.vector_mechanisms.ready_comparison import digest, verified_packet
from scripts.qualify_grain_source_history import load_observer
from scripts.run_grain_vector_mechanism_controls import write_json

CODE = (
    "scripts/collect_grain_source_history.py",
    "scripts/qualify_grain_source_history.py",
    "grain_research/vector_mechanisms/history_collection.py",
    "grain_research/vector_mechanisms/history_instrument.py",
    "grain_research/vector_mechanisms/source_history.py",
    "grain_research/vector_mechanisms/trace.py",
    "grain_research/vector_mechanisms/external_material.py",
    "grain_research/vector_mechanisms/context.py",
)


def replay_qualification(features, metrics, case_index, parent):
    """Compare every captured state and both raw candidate logits against real qualification."""
    errors = []
    for name, value in features.items():
        arm, task, h, site = name.split("__")
        for vi in range(3):
            key = f"case{case_index}_view{vi}_{arm}_{task}_{h}_{site}"
            errors.append(float(np.max(abs(value[vi] - parent[key]))))
    for r in metrics["readouts"]:
        key = f"case{case_index}_view{r['view']}_{r['text_arm']}_{r['task']}_{r['history_arm']}_logits"
        for token, name in metrics["candidate_tokens"].items():
            errors.append(abs(r[name] - float(parent[key][int(token)])))
    if not errors or not np.isfinite(errors).all() or max(errors) != 0:
        raise ValueError(
            "Production collector differs from qualified native observations"
        )
    return len(errors), max(errors)


def require_technical(folder, qm, material, code):
    """Only exact model/code/material replay permits nontechnical extraction."""
    if folder is None:
        raise ValueError("Full history collection needs a complete production replay")
    m, s = verified_packet(folder)
    if (
        m["protocol"] != "VM05_HISTORY_COLLECTION_V1"
        or m["mode"] != "qualification_replay"
        or m["qualification_identity"] != qm["identity"]
        or m["material_identity"] != material["identity"]
        or m["code_sha256"] != code
        or s["authors"] != 6
        or s["token_layouts_replayed"] != 324
        or s["qualified_state_replay_comparison_groups"] != 2268
        or s["qualified_state_replay_max"] != 0
        or not s["real_neural_collection"]
        or s["labels_read"]
        or s["bridge_outcomes_read"]
    ):
        raise ValueError("Invalid technical production replay")
    return m | {"model_signature": s["model_signature"]}


def run(a):
    """A metadata preflight cannot be promoted to a model run or an external result."""
    if not 60 <= a.max_seconds <= 3600 or not 0 < a.max_cache_bytes <= 1_000_000_000:
        raise ValueError("Require bounded single-shard runtime and cache")
    started = time.monotonic()
    by_author, whole, material, audit = registered_cases(
        a.views, a.material_dir, a.audit_dir
    )
    code = {name: digest(ROOT / name) for name in CODE}
    tokenizer_matches(a.qwen_tokenizer, material)
    a.tokenizer = AutoTokenizer.from_pretrained(
        str(a.qwen_tokenizer), local_files_only=True, trust_remote_code=False
    )
    qm = qmodel = technical = None
    if not a.preflight_only:
        if a.qualification_dir is None:
            raise ValueError("No real history qualification")
        qm, _, qmodel = qualification_packet(a.qualification_dir, material, audit, ROOT)
    if a.qualification_replay:
        if a.preflight_only or a.shard_count != 128 or a.shard_index != 0:
            raise ValueError(
                "Technical replay is fixed six-author material, not a shard"
            )
        cases = qm["selected_cases"]
        allowed = {c["author_id"]: c for c in whole}
        if len(cases) != 6 or any(c != allowed.get(c["author_id"]) for c in cases):
            raise ValueError("Qualification cases escaped strict history registry")
        mode = "qualification_replay"
    else:
        cases = shard_material(whole, a.shard_count, a.shard_index)
        mode = "preflight" if a.preflight_only else "source_shard"
        if not a.preflight_only:
            technical = require_technical(a.technical_parent, qm, material, code)
    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Private results directory required")
    out.mkdir(parents=True, exist_ok=False)
    archive_instrument(ROOT, out / "source_snapshot.zip")
    fields = {
        "protocol": "VM05_HISTORY_COLLECTION_V1",
        "mode": mode,
        "material_identity": material["identity"],
        "audit_identity": audit["identity"],
        "qualification_identity": qm["identity"] if qm else None,
        "technical_identity": technical["identity"] if technical else None,
        "strict_author_commitment": material_hash([b["author_id"] for b in whole]),
        "strict_authors": len(whole),
        "authors": [c["author_id"] for c in cases],
        "case_commitment": material_hash(cases),
        "shard_count": a.shard_count,
        "shard_index": a.shard_index,
        "readout_kinds": KINDS,
        "feature_rows": ["decision", "current_mean", "current_std"],
        "sites": SITES,
        "code_sha256": code,
        "source_snapshot_sha256": digest(out / "source_snapshot.zip"),
        "max_seconds": a.max_seconds,
        "max_cache_bytes": a.max_cache_bytes,
        "labels_read": False,
        "bridge_outcomes_read": False,
        "diagnostic_margin_is_direct_trait_score": False,
    }
    fields["identity"] = material_hash(fields)
    write_json(out / "manifest.json", fields)
    cache = out / "private_cache"
    cache.mkdir()
    records = []
    cache_bytes = 0
    checked = 0
    obs = None
    qstates = None
    qualified_replay = 0
    try:
        if not a.preflight_only:
            obs = load_observer(a, material)
            if observation_signature(obs.manifest()) != observation_signature(qmodel):
                raise ValueError("Actual history observer differs from qualification")
            if technical and technical["model_signature"] != observation_signature(
                qmodel
            ):
                raise ValueError("Production technical model changed")
            write_json(out / "actual_observer.json", obs.manifest())
            torch.cuda.reset_peak_memory_stats()
            if a.qualification_replay:
                qstates = np.load(
                    a.qualification_dir / "private_instrument_states.npz",
                    allow_pickle=False,
                )
        for i, case in enumerate(cases):
            if time.monotonic() - started >= a.max_seconds:
                raise TimeoutError("Preserve partial history shard")
            layouts = material_layouts(case, by_author, a.tokenizer)
            checked += len(layouts)
            if obs is None:
                records.append(
                    {
                        "case_commitment": material_hash(case),
                        "layouts": len(layouts),
                        "unique_layouts": len(
                            {material_hash(r) for r in layouts.values()}
                        ),
                    }
                )
                continue
            arrays, metrics = collect_history_author(
                layouts, obs, deadline=started + a.max_seconds, repeat=i == 0
            )
            metrics["candidate_tokens"] = {
                str(obs.tokenizer.encode(t, add_special_tokens=False)[0]): name
                for t, name in (("Yes", "yes_logit"), ("No", "no_logit"))
            }
            if qstates is not None:
                n, error = replay_qualification(arrays, metrics, i, qstates)
                qualified_replay += n
                if error != 0:
                    raise ValueError("Qualified state mismatch")
            for site in SITES:
                for kind in KINDS:
                    if not np.isfinite(
                        history_channel(arrays, "raw", site, kind)
                    ).all():
                        raise ValueError("Nonfinite declared history contrast")
            stored, aliases = pack_features(arrays)
            if any(
                not np.array_equal(v, unpack_features(stored, aliases)[k])
                for k, v in arrays.items()
            ):
                raise ValueError("Lossless aliases did not reconstruct")
            buf = io.BytesIO()
            np.savez_compressed(buf, **stored)
            key = material_hash(case["author_id"])
            metrics.update(
                author_id=case["author_id"],
                source_group=case["source_group"],
                case_commitment=material_hash(case),
                feature_sha256=hashlib.sha256(buf.getvalue()).hexdigest(),
                feature_aliases=aliases,
                cache_format="lossless_history_aliases_v1",
            )
            if (
                cache_bytes + buf.tell() + len(json.dumps(metrics).encode()) + 1024
                > a.max_cache_bytes
            ):
                raise MemoryError("History cache bound; preserve completed authors")
            (cache / (key + ".npz")).write_bytes(buf.getvalue())
            write_json(cache / (key + ".json"), metrics)
            cache_bytes += (cache / (key + ".npz")).stat().st_size + (
                cache / (key + ".json")
            ).stat().st_size
            records.append(
                {
                    "cache_key": key,
                    "author_id": case["author_id"],
                    "npz_sha256": metrics["feature_sha256"],
                    "json_sha256": digest(cache / (key + ".json")),
                    "unique_layouts": metrics["unique_layouts"],
                }
            )
            write_json(
                out / "run_status.json",
                {
                    "status": "RUNNING",
                    "authors": i + 1,
                    "planned": len(cases),
                    "seconds": time.monotonic() - started,
                    "forwards": obs.calls,
                    "cache_bytes": cache_bytes,
                    "labels_read": False,
                },
            )
        write_json(out / "private_records.json", records)
        # Model metadata is an output, so the original pre-load manifest remains immutable.
        summary = {
            "status": "COMPLETE",
            "parent_identity": fields["identity"],
            "authors": len(records),
            "strict_authors": len(whole),
            "token_layouts_replayed": checked,
            "unique_layouts": sum(r["unique_layouts"] for r in records),
            "real_neural_collection": obs is not None,
            "forwards": obs.calls if obs else 0,
            "qualified_state_replay_comparison_groups": qualified_replay,
            "qualified_state_replay_max": 0.0 if qstates is not None else None,
            "cache_bytes": cache_bytes,
            "seconds": time.monotonic() - started,
            "model_signature": observation_signature(qmodel) if qmodel else None,
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
        if len(records) != len(cases) or checked != 54 * len(cases):
            raise ValueError("Incomplete history shard")
        write_json(out / "summary.json", summary)
        write_json(out / "run_status.json", summary)
        print(
            json.dumps({k: v for k, v in summary.items() if k != "outputs_sha256"}),
            flush=True,
        )
    except Exception as e:
        write_json(
            out / "run_status.json",
            {
                "status": "FAILED_STOP",
                "exception_type": type(e).__name__,
                "error": str(e),
                "authors": len(records),
                "planned": len(cases),
                "forwards": obs.calls if obs else 0,
                "cache_bytes": cache_bytes,
                "labels_read": False,
            },
        )
        raise
    finally:
        if qstates is not None:
            qstates.close()


def main():
    """Preflight on Mac; real runs require completed pinned-CUDA qualifications."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("views", "material-dir", "audit-dir", "qwen-tokenizer", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("qualification-dir", "technical-parent", "observer-queue-dir"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--qualification-replay", action="store_true")
    p.add_argument("--shard-count", type=int, default=128)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--max-seconds", type=int, default=1800)
    p.add_argument("--max-cache-bytes", type=int, default=500_000_000)
    run(p.parse_args())


if __name__ == "__main__":
    main()
