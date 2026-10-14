#!/usr/bin/env python3
"""Prepare source-exclusive material and resumable local E5/IPIP evidence caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.predecessors import first_person_sentences, parse_ipip_neo
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_controls import exclude_reference
from grain_research.vector_mechanisms.external_heads import load_source_labels
from grain_research.vector_mechanisms.external_material import unpack_features
from scripts.run_grain_vector_mechanism_baselines import MODELS

BASE = ROOT / "results/grain_vector_mechanisms"
DEFAULT = BASE / "predecessor_integration_v1/material"
IPIP_URL = "https://ipip.ori.org/newNEOKey.htm"


def digest(path):
    """Hash one local provenance artifact."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    """Write JSON atomically for resumable preparation."""
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temp.replace(path)


def prepare(args):
    """Reuse only the frozen Big5-only source material, never bridge or MEPS."""
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    complete = out / "material_manifest.json"
    if complete.exists():
        manifest = json.loads(complete.read_text())
        for name, checksum in manifest["files"].items():
            if digest(out / name) != checksum:
                raise ValueError("Prepared material changed: " + name)
        return manifest
    if any(out.iterdir()):
        raise ValueError("Partial material preparation: use a new output directory")
    parent_dir, preflight = (
        BASE / "vm05_full_source_cohort_r1",
        BASE / "vm05_preflight_r1",
    )
    common = json.loads((preflight / "common_material_manifest.json").read_text())
    parent = json.loads((parent_dir / "manifest.json").read_text())
    bundles = [
        b
        for b in json.loads((parent_dir / "big5_only_author_views.json").read_text())
        if b["author_id"] in set(common["author_ids"])
    ]
    bundles = exclude_reference(
        bundles, pd.read_csv(preflight / "budget_by_author.csv")
    )
    if any(b["source_group"] != "big5_only" for b in bundles):
        raise ValueError("Source-only material required")
    if len({b["author_id"] for b in bundles}) != len(bundles):
        raise ValueError("Duplicate author")
    if args.limit:
        # Diagnostic pilot only; preserve representation of all official folds.
        bundles = [
            b
            for f in range(5)
            for b in sorted(
                (b for b in bundles if b["official_folds"]["big5"] == f),
                key=lambda b: b["author_id"],
            )[: args.limit]
        ]
    queue = BASE / "vm05_full_vectors_r1/e5_large"
    paths, sources = {}, {}
    for shard in sorted(queue.glob("shard_*")):
        manifest = json.loads((shard / "manifest.json").read_text())
        if manifest["full_material_sha256"] != common["material_sha256"]:
            raise ValueError("Encoder/common material mismatch")
        if manifest["model"]["revision"] != MODELS["e5_large"][1]:
            raise ValueError("Encoder revision mismatch")
        for uid in manifest["authors"]:
            if uid in paths:
                raise ValueError("Duplicate encoder author")
            paths[uid] = shard / "private_cache" / uid
    all_vectors, sentences, documents, half_a, half_b, folds, provenance = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for bi, bundle in enumerate(bundles):
        path = paths[bundle["author_id"]]
        meta = json.loads(path.with_suffix(".json").read_text())
        checksum = digest(path.with_suffix(".npz"))
        if (
            checksum != meta["feature_sha256"]
            or meta["author_id"] != bundle["author_id"]
        ):
            raise ValueError("Encoder cache checksum/identity mismatch")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
            features = unpack_features(
                {k: z[k] for k in z.files}, meta.get("feature_aliases", {})
            )
        vectors = features["raw_comment_vectors"].reshape(36, -1)
        comments = [c for view in bundle["views"] for c in view]
        if len(comments) != 36:
            raise ValueError("Expected complete 36-comment author material")
        all_vectors.append(vectors)
        documents.append("\n\n".join(c["body"] for c in comments))
        half_a.append("\n\n".join(c["body"] for c in comments[::2]))
        half_b.append("\n\n".join(c["body"] for c in comments[1::2]))
        folds.append(bundle["official_folds"]["big5"])
        provenance.append({"author_id": bundle["author_id"], "cache_sha256": checksum})
        for ci, c in enumerate(comments):
            for lo, hi, text in first_person_sentences(c["body"]):
                sentences.append(
                    {
                        "author_index": bi,
                        "comment_index": ci,
                        "comment_id": c["comment_id"],
                        "start": lo,
                        "end": hi,
                        "text": text,
                    }
                )
    # All rule-defined material is fixed before loading selected development labels.
    y, label_audit = load_source_labels(
        args.prepared_root, bundles, "big5", parent["source_tables"]
    )
    with np.load(preflight / "grain_frozen_features.npz", allow_pickle=False) as z:
        ix = {str(u): i for i, u in enumerate(z["author_ids"])}
        grain = z["all36_raw_features"][[ix[b["author_id"]] for b in bundles]]
    write(
        out / "private_material.json",
        {
            "bundles": bundles,
            "documents": documents,
            "half_a": half_a,
            "half_b": half_b,
            "sentences": sentences,
            "source_provenance": provenance,
        },
    )
    np.savez_compressed(
        out / "private_arrays.npz",
        author_ids=np.asarray([b["author_id"] for b in bundles]),
        folds=folds,
        y=y,
        grain=grain,
        comments=np.stack(all_vectors),
    )
    import requests

    response = requests.get(IPIP_URL, timeout=30)
    response.raise_for_status()
    page = response.content.decode("windows-1252")
    items = parse_ipip_neo(page)
    write(
        out / "ipip_neo300.json",
        {
            "source_url": IPIP_URL,
            "source_sha256": hashlib.sha256(response.content).hexdigest(),
            "permission_url": "https://ipip.ori.org/newPermission.htm",
            "license": "public_domain",
            "items": items,
        },
    )
    files = ("private_material.json", "private_arrays.npz", "ipip_neo300.json")
    sources["common_manifest"] = digest(preflight / "common_material_manifest.json")
    sources["parent_manifest"] = digest(parent_dir / "manifest.json")
    manifest = {
        "protocol": "GRAIN_PREDECESSOR_MATERIAL_V1",
        "authors": len(bundles),
        "comments": 36 * len(bundles),
        "sentences": len(sentences),
        "items": len(items),
        "source_only": True,
        "meps_used": False,
        "pilot_per_fold": args.limit,
        "label_audit": label_audit,
        "model": MODELS["e5_large"],
        "material_sha256": material_hash(bundles),
        "sources": sources,
        "files": {name: digest(out / name) for name in files},
    }
    write(complete, manifest)
    print(
        {k: manifest[k] for k in ("authors", "comments", "sentences", "items")},
        flush=True,
    )
    return manifest


def encode(args, manifest):
    """Encode only new item/sentence evidence locally; save each resumable batch."""
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    out = args.output
    data = json.loads((out / "private_material.json").read_text())
    items = json.loads((out / "ipip_neo300.json").read_text())["items"]
    texts = [x["text"] for x in items] + [s["text"] for s in data["sentences"]]
    name, revision, prefix, _ = MODELS["e5_large"]
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(
        name, revision=revision, local_files_only=True, trust_remote_code=False
    )
    tokens = tokenizer([prefix + t for t in texts], truncation=False)["input_ids"]
    if max(map(len, tokens)) > 512:
        raise ValueError(
            "Sentence exceeds cache-compatible token cap; do not silently truncate"
        )
    cache = out / "sentence_batches"
    cache.mkdir(exist_ok=True)
    contract = {
        "material_manifest_sha256": digest(out / "material_manifest.json"),
        "model": name,
        "revision": revision,
        "prefix": prefix,
        "pool": "masked_mean_l2",
        "dtype": "float32",
        "device": args.device,
        "batch_size": args.batch_size,
        "padding": "longest_in_batch",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "texts_sha256": material_hash(texts),
        "count": len(texts),
        "max_tokens": max(map(len, tokens)),
    }
    path = cache / "contract.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("Changed resumable encoder contract")
    write(path, contract)
    model = None
    started, completed = time.monotonic(), 0
    for start in range(0, len(texts), args.batch_size):
        stop = min(start + args.batch_size, len(texts))
        target = cache / f"batch_{start:06d}.npz"
        meta = target.with_suffix(".json")
        if target.exists() and meta.exists():
            info = json.loads(meta.read_text())
            if (
                info["sha256"] != digest(target)
                or info["start"] != start
                or info["stop"] != stop
            ):
                raise ValueError("Changed batch")
            completed = stop
            continue
        if time.monotonic() - started > args.max_seconds:
            write(
                out / "encoding_status.json",
                {
                    "status": "PAUSED_BUDGET",
                    "completed": completed,
                    "total": len(texts),
                },
            )
            return
        if model is None:
            model = (
                AutoModel.from_pretrained(
                    name,
                    revision=revision,
                    local_files_only=True,
                    trust_remote_code=False,
                    dtype=torch.float32,
                )
                .eval()
                .to(args.device)
            )
        batch = tokenizer(
            [prefix + t for t in texts[start:stop]],
            padding=True,
            truncation=False,
            return_tensors="pt",
        ).to(args.device)
        with torch.inference_mode():
            h = model(**batch).last_hidden_state
            mask = batch["attention_mask"][:, :, None]
            vectors = (
                torch.nn.functional.normalize((h * mask).sum(1) / mask.sum(1), dim=1)
                .cpu()
                .numpy()
            )
        if not np.isfinite(vectors).all():
            raise ValueError("Nonfinite sentence embeddings")
        np.savez_compressed(target, vectors=vectors)
        write(meta, {"start": start, "stop": stop, "sha256": digest(target)})
        completed = stop
        if start % (args.batch_size * 20) == 0:
            print(
                {
                    "encoded": completed,
                    "total": len(texts),
                    "seconds": round(time.monotonic() - started, 1),
                },
                flush=True,
            )
    write(
        out / "encoding_status.json",
        {
            "status": "COMPLETE",
            "completed": completed,
            "total": len(texts),
            "contract_sha256": digest(path),
            "seconds_this_invocation": time.monotonic() - started,
        },
    )


def main():
    """Run a bounded, local-only preparation pass with explicit output location."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT)
    p.add_argument(
        "--prepared-root",
        type=Path,
        default=ROOT.parent / "project persona/data_sets/prepared/pandora_official",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Diagnostic authors per fold; 0 uses all source authors",
    )
    p.add_argument("--encode", action="store_true")
    p.add_argument("--device", choices=["cpu", "mps"], default="mps")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-seconds", type=int, default=1800)
    a = p.parse_args()
    if a.limit < 0 or a.batch_size < 1 or a.max_seconds < 1:
        p.error("Invalid resource bound")
    manifest = prepare(a)
    if a.encode:
        encode(a, manifest)


if __name__ == "__main__":
    main()
