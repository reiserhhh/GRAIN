#!/usr/bin/env python3
"""Bounded SIMPA-family paraphrase reference, keyed counts and semantic error probes.

The paper names a RoBERTa paraphrase family, not a verifiable checkpoint revision.
This pinned public model is a family-faithful adapter, never exact replication.
No new network training and no personality-label-based threshold selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.predecessors import mean_pearson
from grain_research.vector_mechanisms.external_heads import metrics
from scripts.prepare_grain_predecessor_reference import BASE, DEFAULT, digest, write
from scripts.run_grain_predecessor_reference import (
    evidence_features,
    fit_fold,
    paired_delta,
    read_material,
)

MODEL = "sentence-transformers/paraphrase-distilroberta-base-v1"
REVISION = "48bffbbd27bf028ecdd0cd55abb51236ec12ef1b"
E5_REVISION = "f169b11e22de13617baa190a028a32f3493550b6"


def contrasts():
    """Author-designed semantic fixtures, not human-validated natural-data labels.

    The subject-shift examples retain 'I' but attribute the property to a friend;
    episodic examples must not automatically inherit the habitual trait reading.
    """
    examples = [
        (
            "I make friends easily.",
            "Making new friends comes naturally to me.",
            "I find it very difficult to make friends.",
            "I noticed that my friend makes friends easily.",
            "I made a new friend today.",
        ),
        (
            "I keep others at a distance.",
            "I tend to hold people at arm's length.",
            "I generally let people get close to me.",
            "I noticed that my friend keeps others at a distance.",
            "I avoided a stranger yesterday.",
        ),
        (
            "I get stressed out easily.",
            "Small pressures quickly make me feel stressed.",
            "It takes a lot of pressure before I feel stressed.",
            "I noticed that my friend gets stressed out easily.",
            "I felt stressed during an accident.",
        ),
        (
            "I am relaxed most of the time.",
            "I usually feel calm and at ease.",
            "I rarely feel relaxed or at ease.",
            "I noticed that my friend is relaxed most of the time.",
            "I felt relaxed during my break.",
        ),
        (
            "I insult people.",
            "I frequently make insulting remarks to people.",
            "I never insult other people.",
            "I noticed that my friend insults people.",
            "I insulted someone once during an argument.",
        ),
        (
            "I sympathize with the homeless.",
            "I feel compassion for people without a home.",
            "I feel no sympathy for people without a home.",
            "I noticed that my friend sympathizes with the homeless.",
            "I helped a homeless person today.",
        ),
        (
            "I am always prepared.",
            "I habitually get everything ready in advance.",
            "I am hardly ever prepared.",
            "I noticed that my friend is always prepared.",
            "I was prepared for today's meeting.",
        ),
        (
            "I leave my belongings around.",
            "I tend to leave my possessions lying everywhere.",
            "I always put my belongings away.",
            "I noticed that my friend leaves belongings around.",
            "I left a book on the floor today.",
        ),
        (
            "I have a vivid imagination.",
            "I can easily imagine rich and detailed scenes.",
            "I do not have a vivid imagination.",
            "I noticed that my friend has a vivid imagination.",
            "I imagined a scene during that game.",
        ),
        (
            "I am not interested in abstract ideas.",
            "Abstract concepts generally do not interest me.",
            "I am fascinated by abstract ideas.",
            "I noticed that my friend is not interested in abstract ideas.",
            "I skipped an abstract topic today.",
        ),
    ]
    categories = ("equivalent", "opposite", "other_person", "single_episode")
    return [
        {
            "target": row[0],
            "sentence": text,
            "category": category,
            "exact_habitual_self_match_expected": category == "equivalent",
        }
        for row in examples
        for category, text in zip(categories, row[1:], strict=True)
    ]


def encode_batches(
    texts,
    cache,
    model_name,
    revision,
    *,
    prefix="",
    cap=128,
    allow_download=False,
    max_seconds=900,
):
    """Encode locally with pinned safetensors and resumable checksum-bound batches."""
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(4)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    path = snapshot_download(
        model_name,
        revision=revision,
        local_files_only=not allow_download,
        allow_patterns=["*.json", "*.txt", "model.safetensors", "1_Pooling/*"],
    )
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    lengths = [
        len(t)
        for t in tokenizer([prefix + t for t in texts], truncation=False)["input_ids"]
    ]
    contract = {
        "model": model_name,
        "revision": revision,
        "prefix": prefix,
        "cap": cap,
        "pooling": "attention_mask_mean_l2",
        "dtype": "float32",
        "device": device,
        "batch_size": 32,
        "padding": "longest_in_batch",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "texts": len(texts),
        "texts_sha256": hashlib.sha256(
            json.dumps(texts, ensure_ascii=False).encode()
        ).hexdigest(),
        "truncated_texts": sum(n > cap for n in lengths),
        "maximum_tokens": max(lengths),
    }
    cache.mkdir(parents=True, exist_ok=True)
    cpath = cache / "contract.json"
    if cpath.exists() and json.loads(cpath.read_text()) != contract:
        raise ValueError("Encoder resume contract changed")
    write(cpath, contract)
    began, vectors, model = time.monotonic(), [], None
    for start in range(0, len(texts), 512):
        stop = min(start + 512, len(texts))
        target = cache / f"batch_{start:06d}.npz"
        metadata = target.with_suffix(".json")
        if target.exists() and metadata.exists():
            meta = json.loads(metadata.read_text())
            if meta != {"start": start, "stop": stop, "sha256": digest(target)}:
                raise ValueError("Encoded batch changed")
            with np.load(target, allow_pickle=False) as z:
                v = z["vectors"]
        else:
            if time.monotonic() - began > max_seconds:
                write(
                    cache / "status.json",
                    {
                        "status": "PAUSED_BUDGET",
                        "completed": start,
                        "total": len(texts),
                    },
                )
                raise TimeoutError(
                    "Encoding budget reached; same command resumes cached batches"
                )
            if model is None:
                model = (
                    AutoModel.from_pretrained(
                        path,
                        trust_remote_code=False,
                        use_safetensors=True,
                        dtype=torch.float32,
                    )
                    .eval()
                    .to(device)
                )
            blocks = []
            for i in range(start, stop, 32):
                batch = tokenizer(
                    [prefix + t for t in texts[i : min(i + 32, stop)]],
                    padding=True,
                    truncation=True,
                    max_length=cap,
                    return_tensors="pt",
                ).to(device)
                with torch.inference_mode():
                    h = model(**batch).last_hidden_state
                    mask = batch["attention_mask"][:, :, None]
                    block = torch.nn.functional.normalize(
                        (h * mask).sum(1) / mask.sum(1), dim=1
                    )
                    blocks.append(block.cpu().numpy())
            v = np.concatenate(blocks)
            np.savez_compressed(target, vectors=v)
            write(metadata, {"start": start, "stop": stop, "sha256": digest(target)})
        if len(v) != stop - start or not np.isfinite(v).all():
            raise ValueError("Invalid encoded block")
        vectors.append(v)
        if start % 4096 == 0:
            print(
                json.dumps(
                    {
                        "encoder": model_name,
                        "encoded": stop,
                        "total": len(texts),
                        "seconds": round(time.monotonic() - began, 1),
                    }
                ),
                flush=True,
            )
    write(
        cache / "status.json",
        {
            "status": "COMPLETE",
            "completed": len(texts),
            "contract_sha256": digest(cpath),
        },
    )
    del model
    if device == "mps":
        torch.mps.empty_cache()
    return np.concatenate(vectors), contract


def detection_geometry(vectors, n_items):
    """Report selectivity, not false-positive rate: random pairs can be relevant."""
    q, h = vectors[:n_items], vectors[n_items:]
    sim = h @ q.T
    top2 = np.partition(sim, -2, axis=1)[:, -2:]
    rng = np.random.default_rng(731)
    null = sim[np.arange(len(h)), rng.integers(0, len(q), len(h))]
    return {
        "top1_acceptance_at_0_6": float(np.mean(top2[:, 1] >= 0.6)),
        "random_pair_acceptance_at_0_6": float(np.mean(null >= 0.6)),
        "top1_quantiles": np.quantile(top2[:, 1], [0.05, 0.5, 0.95]).tolist(),
        "margin_quantiles": np.quantile(
            top2[:, 1] - top2[:, 0], [0.05, 0.5, 0.95]
        ).tolist(),
    }


def run(args):
    """Compare a source-motivated encoder correction, not a new threshold search."""
    out = args.output
    if (out / "summary.json").exists():
        raise ValueError("Completed outputs are immutable; choose a fresh directory")
    out.mkdir(parents=True, exist_ok=True)
    manifest, arrays, data, items, e5 = read_material(args.material)
    expected = {
        "source_manifest": digest(args.material / "material_manifest.json"),
        "model": MODEL,
        "revision": REVISION,
        "threshold": 0.6,
        "code_sha256": digest(Path(__file__)),
        "scope": "SIMPA_family_adapter_not_exact_native_checkpoint",
    }
    contract = out / "run_contract.json"
    if contract.exists() and json.loads(contract.read_text()) != expected:
        raise ValueError("Run contract changed")
    write(contract, expected)
    texts = [i["text"] for i in items] + [s["text"] for s in data["sentences"]]
    vectors, encoder = encode_batches(
        texts,
        out / "paraphrase_cache",
        MODEL,
        REVISION,
        allow_download=args.allow_download,
    )
    geometry = {
        "e5": detection_geometry(e5, len(items)),
        "paraphrase": detection_geometry(vectors, len(items)),
    }
    write(out / "detection_geometry.json", geometry)
    fixtures = contrasts()
    public = [r[k] for r in fixtures for k in ("target", "sentence")]
    probe_rows = []
    for name, model, revision, prefix, cap in (
        ("paraphrase", MODEL, REVISION, "", 128),
        ("e5", "intfloat/e5-large-v2", E5_REVISION, "query: ", 512),
    ):
        v, _ = encode_batches(
            public,
            out / (name + "_fixture_cache"),
            model,
            revision,
            prefix=prefix,
            cap=cap,
        )
        for i, record in enumerate(fixtures):
            similarity = float(v[i * 2] @ v[i * 2 + 1])
            probe_rows.append(
                {
                    "encoder": name,
                    **record,
                    "similarity": similarity,
                    "pair_above_0_6": similarity >= 0.6,
                }
            )
    probes = pd.DataFrame(probe_rows)
    probes.to_csv(out / "public_contrast_probes.csv", index=False)
    probes.groupby(["encoder", "category"]).agg(
        mean_similarity=("similarity", "mean"),
        acceptance=("pair_above_0_6", "mean"),
        pairs=("similarity", "size"),
    ).reset_index().to_csv(out / "contrast_summary.csv", index=False)
    counts, halves, evidence = evidence_features(
        data, items, vectors, len(arrays["y"]), 0.6
    )
    evidence.to_csv(out / "private_sentence_evidence.csv", index=False)
    np.savez_compressed(out / "private_counts.npz", counts=counts, halves=halves)
    # Only discovery-side sentences enter the future annotation/calibration packet.
    source_sentences = [
        i
        for i, s in enumerate(data["sentences"])
        if arrays["folds"][s["author_index"]] != 0
    ]
    rng = np.random.default_rng(731)
    audit = []
    for i in rng.choice(
        source_sentences, min(80, len(source_sentences)), replace=False
    ):
        sim = vectors[len(items) + i] @ vectors[: len(items)].T
        j = int(sim.argmax())
        s = data["sentences"][i]
        audit.append(
            {
                "sentence_index": int(i),
                "sentence": s["text"],
                "item_id": items[j]["id"],
                "item_text": items[j]["text"],
                "comment_id": s["comment_id"],
                "start": s["start"],
                "end": s["end"],
                "self_attribution": "",
                "simpa_category_1_to_7": "",
                "rater": "",
                "rater_kind": "",
                "similarity_hidden": "",
            }
        )
    pd.DataFrame(audit).to_csv(
        out / "private_source_annotation_packet.csv", index=False
    )
    sources = {
        "documents": np.asarray(data["documents"]),
        "grain": arrays["grain"],
        "semantic": arrays["comments"].mean(1),
        "items": counts.reshape(len(counts), -1),
    }
    y, folds = arrays["y"], arrays["folds"]
    predictions, rows, choices = {}, [], []
    for method in ("item_counts_ridge", "semantic_items", "grain_items"):
        pred = np.full_like(y, np.nan)
        for fold in range(5):
            train, held = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            feature, head, p, selected = fit_fold(sources, y, train, held, method)
            pred[held] = p
            joblib.dump(
                {"feature": feature, "head": head, "source_indices": train},
                out / f"private_{method}_fold{fold}.joblib",
            )
            choices.append({"method": method, "fold": fold, **selected})
        predictions[method] = pred
        rows.extend({"method": method, **m} for m in metrics(y, pred, "big5"))
        print(
            json.dumps({"method": method, "mean_r": mean_pearson(y, pred)}), flush=True
        )
    pd.DataFrame(rows).to_csv(out / "metrics_by_trait.csv", index=False)
    write(out / "inner_selections.json", choices)
    np.savez_compressed(
        out / "private_oof.npz",
        author_ids=arrays["author_ids"],
        y=y,
        folds=folds,
        **predictions,
    )
    with np.load(args.comparison / "private_oof.npz", allow_pickle=False) as original:
        if not np.array_equal(
            original["author_ids"], arrays["author_ids"]
        ) or not np.array_equal(original["y"], y):
            raise ValueError("Original comparison author/label mismatch")
        deltas = {
            method: paired_delta(y, pred, original[method])
            for method, pred in predictions.items()
        }
        deltas["semantic_items_vs_semantic_baseline"] = paired_delta(
            y, predictions["semantic_items"], original["semantic_pca_ridge"]
        )
    result = {
        "status": "COMPLETE",
        "authors": manifest["authors"],
        "sentences": manifest["sentences"],
        "model": MODEL,
        "revision": REVISION,
        "encoder_contract": encoder,
        "mean_pearson": {k: mean_pearson(y, p) for k, p in predictions.items()},
        "paired_deltas": deltas,
        "geometry": geometry,
        "human_annotation_status": "NOT_ANNOTATED; 80 source-only blind cases prepared",
        "threshold_status": "0.6 frozen as paper-family reference, not validated precision calibration",
        "contrast_scope": "40 designed semantic pairs, not natural-data sensitivity/specificity",
    }
    write(out / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    """Use --allow-download for the pinned public paraphrase weights only."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--material", type=Path, default=DEFAULT)
    p.add_argument(
        "--output", type=Path, default=BASE / "predecessor_integration_v1/paraphrase_r1"
    )
    p.add_argument(
        "--comparison",
        type=Path,
        default=BASE / "predecessor_integration_v1/comparison_r2",
    )
    p.add_argument("--allow-download", action="store_true")
    with threadpool_limits(limits=4):
        run(p.parse_args())


if __name__ == "__main__":
    main()
