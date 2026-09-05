#!/usr/bin/env python3
"""Compare independent-comment traces with frozen and embedding-first baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.comparison import (
    describe_representation,
    node_bootstrap_weights,
    shuffled_author_null,
)
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from scripts.run_grain_vector_mechanism_controls import write_json

MODELS = {
    "bge_large": (
        "BAAI/bge-large-en-v1.5",
        "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        "",
        "cls",
    ),
    "e5_large": (
        "intfloat/e5-large-v2",
        "f169b11e22de13617baa190a028a32f3493550b6",
        "query: ",
        "mean",
    ),
}


def encode_local(texts, spec, device):
    """Use official pooling/prefix, fixed FP32 shape, and no network or truncation."""
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    name, revision, prefix, pool = spec
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(
        name, revision=revision, local_files_only=True, trust_remote_code=False
    )
    prepared = [prefix + t for t in texts]
    lengths = [len(tokenizer.encode(t, add_special_tokens=True)) for t in prepared]
    if max(lengths) > 512:
        raise ValueError(
            f"Equal material exceeds {name} token limit; do not truncate silently"
        )
    model = (
        AutoModel.from_pretrained(
            name,
            revision=revision,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        .eval()
        .to(device)
    )
    vectors = []
    for start in range(0, len(texts), 4):
        batch = tokenizer(
            prepared[start : start + 4],
            padding="max_length",
            max_length=512,
            truncation=False,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            h = model(**batch).last_hidden_state
            if pool == "cls":
                vector = h[:, 0]
            else:
                mask = batch["attention_mask"][:, :, None]
                vector = (h * mask).sum(dim=1) / mask.sum(dim=1)
            vectors.append(torch.nn.functional.normalize(vector, dim=1).cpu().numpy())
    result = np.concatenate(vectors)
    if not np.isfinite(result).all():
        raise ValueError("Encoder returned nonfinite states")
    manifest = {
        "name": name,
        "revision": revision,
        "prefix": prefix,
        "pool": pool,
        "dtype": "float32",
        "device": device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "max_observed_tokens": max(lengths),
        "fixed_padding": 512,
        "batch_size": 4,
        "truncated": 0,
    }
    del model
    if device == "mps":
        torch.mps.empty_cache()
    return result, manifest


def style_features(text):
    """Small format-only comparator, not a personality scoring rule."""
    words = re.findall(r"\b\w+\b", text)
    n = max(1, len(words))
    return [
        np.log1p(n),
        np.log1p(len(text)),
        np.mean([len(w) for w in words]) if words else 0,
        sum(c.isupper() for c in text) / max(1, len(text)),
        text.count("?") / n,
        text.count("!") / n,
        text.count("\n") / n,
        text.count('"') / n,
        len(re.findall(r"https?://", text)) / n,
        sum(c.isdigit() for c in text) / max(1, len(text)),
    ]


def main():
    """Reuse exact parent author/comment IDs and score the full unselected layer grid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--skip-encoders", action="store_true")
    args = parser.parse_args()
    source = json.loads(args.views.read_text(encoding="utf-8"))
    parent = json.loads((args.parent_dir / "manifest.json").read_text())
    if (
        material_hash(source) != parent["source_view_sha256"]
        or parent["history_design"] != "three_nonoverlapping_single_comments"
    ):
        raise ValueError("Parent is not the same independent-history material")
    if (
        json.loads((args.parent_dir / "run_status.json").read_text())["status"]
        != "COMPLETE"
    ):
        raise ValueError("Incomplete parent")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_status.json").exists():
        raise ValueError("Use a new baseline output directory")
    by_id = {b["author_id"]: b for b in source}
    authors = sorted({c["author_id"] for c in parent["cases"]})
    refs = sorted(b["author_id"] for b in source if b["split"] == "technical_only")
    if set(authors) & set(refs):
        raise ValueError("Reference/evaluation authors overlap")
    rows = [
        {
            "author_id": uid,
            "history_index": i,
            "split": "reference" if uid in refs else "pilot",
            **by_id[uid]["history"][i],
        }
        for uid in refs + authors
        for i in range(3)
    ]
    texts = [r["body"] for r in rows]
    train = np.array([r["split"] == "reference" for r in rows])
    source_rows = [
        {k: v for k, v in r.items() if k != "body"}
        | {"text_sha256": hashlib.sha256(r["body"].encode()).hexdigest()}
        for r in rows
    ]
    write_json(output / "input_ledger.json", source_rows)
    manifest = {
        "protocol": "GRAIN_VM02_SAME_COMMENT_BASELINES_V1",
        "parent_identity": parent["identity"],
        "source_view_sha256": material_hash(source),
        "reference_authors": len(refs),
        "pilot_authors": len(authors),
        "code_sha256": code_fingerprint(ROOT),
        "encoder_specs": MODELS,
        "skip_encoders": args.skip_encoders,
        "labels_used": False,
    }
    manifest["identity"] = material_hash(manifest)
    write_json(output / "manifest.json", manifest)
    archive_source(ROOT, output / "source_snapshot.zip")
    started = time.monotonic()
    features, provenance = {}, {}
    for analyzer, ngram in (("word", (1, 2)), ("char", (2, 5))):
        for fit_mode in ("reference_only", "transductive_unlabeled"):
            vectorizer = TfidfVectorizer(
                analyzer=analyzer,
                ngram_range=ngram,
                max_features=20000,
                min_df=1,
                sublinear_tf=True,
            )
            vectorizer.fit(
                np.array(texts)[train] if fit_mode == "reference_only" else texts
            )
            key = f"tfidf_{analyzer}_{fit_mode}"
            features[key] = vectorizer.transform(texts).toarray().astype(np.float32)
            provenance[key] = {
                "fit": fit_mode,
                "dimensions": len(vectorizer.vocabulary_),
                "meaning": "raw text lexical comparator",
            }
        hashing = HashingVectorizer(
            analyzer=analyzer,
            ngram_range=ngram,
            n_features=8192,
            alternate_sign=False,
            norm="l2",
        )
        features[f"hash_{analyzer}"] = (
            hashing.transform(texts).toarray().astype(np.float32)
        )
        provenance[f"hash_{analyzer}"] = {"fit": "none", "dimensions": 8192}
    from scripts.suica_v4_lib import ANCHOR_LEXICONS_V4, score_slices_v4

    rates = score_slices_v4(pd.DataFrame({"slice_text": texts}))
    columns23 = sorted(f"{name}_rate" for name in ANCHOR_LEXICONS_V4)
    columns5 = [
        "novelty_play_v2",
        "directive_action_v2",
        "adversity_recovery_v2",
        "first_person_usage_v2",
        "tension_core_v2",
    ]
    for key, values in (
        ("legacy_anchor23", rates[columns23].to_numpy()),
        ("legacy_construct5", rates[columns5].to_numpy()),
        ("format10", np.asarray([style_features(t) for t in texts])),
    ):
        scaler = StandardScaler().fit(values[train])
        features[key] = scaler.transform(values)
        provenance[key] = {
            "fit": "reference_only_scaler",
            "scope": "feature baseline, NOT full GRAIN author scoring",
        }
    path = (
        ROOT
        / "results/v7_multiview_projection/e1_v72_full_20260715/artifacts/common_source_comment_representation.joblib"
    )
    if path.exists():
        runtime = joblib.load(path)
        features["grain_frozen_unit_projection"] = runtime.transform(texts)
        provenance["grain_frozen_unit_projection"] = {
            "artifact": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "spec": runtime.spec.to_dict(),
            "scope": "frozen unit transform only; historical PANDORA exposure; NOT full GRAIN author score",
        }
    else:
        provenance["grain_frozen_unit_projection"] = {
            "status": "SKIPPED_MISSING_ARTIFACT"
        }
    for key, spec in MODELS.items():
        if args.skip_encoders:
            provenance[key] = {"status": "EXPLICIT_SKIP"}
            continue
        features[key], provenance[key] = encode_local(texts, spec, args.device)
        np.savez_compressed(output / f"{key}_cache.npz", vectors=features[key])
        print(
            f"ENCODED {key} texts={len(texts)} elapsed={time.monotonic() - started:.1f}",
            flush=True,
        )
    pilot_features = {
        k: v[~train].reshape(len(authors), 3, -1) for k, v in features.items()
    }
    for uid_i, uid in enumerate(authors):
        for i in range(3):
            case_id = f"{uid}__c0__history_{i}__behavior"
            case = next(c for c in parent["cases"] if c["case_id"] == case_id)
            if case.get("history_comment_ids") != [
                by_id[uid]["history"][i]["comment_id"]
            ]:
                raise ValueError("LLM cached history is not the comparison comment")
            with np.load(args.parent_dir / "private_cache" / f"{case_id}.npz") as trace:
                for site in trace.files:
                    if site.endswith(("__history_mean", "__decision")):
                        key = "qwen_" + site
                        if key not in pilot_features:
                            pilot_features[key] = np.empty(
                                (len(authors), 3, trace[site].shape[0])
                            )
                        pilot_features[key][uid_i, i] = trace[site]
                        provenance[key] = {
                            "fit": "none",
                            "parent_identity": parent["identity"],
                            "scope": "same history text; Qwen additionally reads fixed task/system prefix",
                        }
    np.savez_compressed(output / "comparison_vectors.npz", **pilot_features)
    write_json(output / "provenance.json", provenance)
    weights = node_bootstrap_weights(len(authors))
    stats, bootstraps = [], {}
    for name, values in pilot_features.items():
        result, bootstrap = describe_representation(values, weights)
        bootstraps[name] = bootstrap
        null = (
            shuffled_author_null(values, draws=199)
            if values.shape[-1] <= 1536
            else {"permutation_p": None, "null_mean": None, "permutations": 0}
        )
        stats.append({"method": name, **result, **null})
    summary = pd.DataFrame(stats).sort_values("author_rank_auc", ascending=False)
    summary.to_csv(output / "comparison.csv", index=False)
    deltas = []
    for candidate in (
        "qwen_block_10__history_mean",
        "qwen_block_16__decision",
        "qwen_final_norm__history_mean",
    ):
        for baseline in (
            "bge_large",
            "e5_large",
            "tfidf_word_reference_only",
            "hash_word",
            "grain_frozen_unit_projection",
            "legacy_anchor23",
        ):
            if baseline not in bootstraps:
                continue
            difference = bootstraps[candidate] - bootstraps[baseline]
            points = summary.set_index("method").author_rank_auc
            deltas.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "delta": points[candidate] - points[baseline],
                    "ci_low": np.quantile(difference, 0.025),
                    "ci_high": np.quantile(difference, 0.975),
                    "scope": "exploratory paired author-node bootstrap; block10 was noticed in same pilot",
                }
            )
    pd.DataFrame(deltas).to_csv(output / "paired_differences.csv", index=False)
    write_json(
        output / "run_status.json",
        {
            "status": "COMPLETE",
            "authors": len(authors),
            "reference_authors": len(refs),
            "comments_compared": len(authors) * 3,
            "methods": len(pilot_features),
            "seconds": time.monotonic() - started,
            "full_grain_score": "NOT_APPLICABLE_ONE_UNIT_PER_VIEW; frozen unit transform and legacy features only",
            "labels_loaded": False,
        },
    )
    print(
        summary[["method", "author_rank_auc", "ci_low", "ci_high"]]
        .round(5)
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
