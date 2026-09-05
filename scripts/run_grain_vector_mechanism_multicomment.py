#!/usr/bin/env python3
"""Build full GRAIN and equal-evidence natural multi-comment baselines."""

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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.cohort import validate_multicomment_bundle
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from grain_research.vector_mechanisms.multicomment import (
    common_groups,
    pool_groups,
    view_layout,
)
from scripts.run_grain_vector_mechanism_baselines import MODELS, style_features
from scripts.run_grain_vector_mechanism_controls import write_json


def grain_scores(texts, bundles, ref_mask, root):
    """Apply frozen projection and complete score/support machinery, never refit it."""
    from suica_core.v7_geometry import GeometryBundle, score_geometry_bundle
    from suica_core.v7_psychometric import author_features_from_embeddings
    from suica_core.v8_bridge import canonical_orbit_distance_signatures
    from suica_core.v8_external_connection import canonical_scale_residual

    runtime_path = (
        root
        / "results/v7_multiview_projection/e1_v72_full_20260715/artifacts/common_source_comment_representation.joblib"
    )
    geometry_path = (
        root / "results/v7_geometry/g1_corrected_v2_full_20260715/geometry_bundle.json"
    )
    runtime = joblib.load(runtime_path)
    geometry = GeometryBundle.from_dict(
        json.loads(geometry_path.read_text(encoding="utf-8"))
    )
    unit_vectors = runtime.transform(texts)
    n = len(bundles)
    unit_vectors = np.asarray(unit_vectors).reshape(n, 36, -1)
    features, statuses, metadata = {}, {}, {}
    reference_scaler = None
    for layout in ("chronological", "interleaved"):
        rows, embeddings = [], []
        for i, b in enumerate(bundles):
            comments = [c for v in b["views"] for c in v]
            for v, group in enumerate(view_layout(b, layout)):
                for index in group:
                    rows.append(
                        {
                            "user_id": f"{b['author_id']}_{v}",
                            "split": "all",
                            "token_count": comments[index]["word_count"],
                        }
                    )
                    embeddings.append(unit_vectors[i, index])
        author = author_features_from_embeddings(
            pd.DataFrame(rows), np.asarray(embeddings)
        )
        raw = author[geometry.feature_names].to_numpy(float)
        if not np.isfinite(raw).all():
            raise ValueError("Unexpected missing full GRAIN author feature")
        score = score_geometry_bundle(
            geometry, raw, unit_counts=author.n_units.to_numpy()
        )
        white = (raw - np.asarray(geometry.feature_center)) @ np.asarray(
            geometry.metric_whitener
        )
        canonical, _, diagnostic = canonical_orbit_distance_signatures(
            white, np.asarray(geometry.reference_landmarks)
        )
        if reference_scaler is None:
            reference_scaler = StandardScaler().fit(raw[ref_mask])
        variants = {
            "unit_mean": raw[:, : unit_vectors.shape[-1]],
            "author_mean_std_reference_scaled": reference_scaler.transform(raw),
            "geometry_sorted": score["landmark_distance_profile"],
            "canonical_profile": canonical,
            "canonical_scale_residual": canonical_scale_residual(canonical),
        }
        # Do not assume the historical feature-name order is all means then stds.
        means = [
            i
            for i, name in enumerate(geometry.feature_names)
            if name.startswith("mean::")
        ]
        variants["unit_mean"] = raw[:, means]
        for method, values in variants.items():
            key = f"grain_{method}__{layout}"
            features[key] = np.asarray(values).reshape(n, 3, -1)
            statuses[key] = (
                np.asarray(score["status"]).reshape(n, 3)
                if method
                in ("geometry_sorted", "canonical_profile", "canonical_scale_residual")
                else np.full((n, 3), "UNRESTRICTED_FEATURE_COMPARATOR")
            )
        metadata[layout] = {
            "status_counts": pd.Series(score["status"]).value_counts().to_dict(),
            "support_radius_threshold": score["support_radius_threshold"],
            "radius_min_median_max": np.quantile(
                score["reference_radius"], [0, 0.5, 1]
            ).tolist(),
            "canonical_diagnostics": diagnostic,
        }
    metadata.update(
        {
            "runtime_sha256": hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
            "geometry_sha256": hashlib.sha256(geometry_path.read_bytes()).hexdigest(),
            "scope": "Frozen full geometry on new raw 12-comment views; not historical cleaned-operator reproduction",
            "comparison_scaler": "Chronological reference only; fixed for interleaving",
        }
    )
    return features, statuses, metadata, unit_vectors


def encode_progress(texts, spec, device, output, *, batch_size=4):
    """Pinned local encoder, fixed padding, finite-state checks, progress cache."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    name, revision, prefix, pool = spec
    tok = AutoTokenizer.from_pretrained(name, revision=revision, local_files_only=True)
    lengths = [len(tok.encode(prefix + t)) for t in texts]
    if max(lengths) > 512:
        raise ValueError("Encoder material exceeds 512; no truncation")
    torch.set_num_threads(4)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = (
        AutoModel.from_pretrained(
            name,
            revision=revision,
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        .eval()
        .to(device)
    )
    out = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
    start = time.monotonic()
    for i in range(0, len(texts), batch_size):
        batch = tok(
            [prefix + t for t in texts[i : i + batch_size]],
            padding="max_length",
            max_length=512,
            truncation=False,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"][:, :, None]
            v = hidden[:, 0] if pool == "cls" else (hidden * mask).sum(1) / mask.sum(1)
            v = torch.nn.functional.normalize(v, dim=1).cpu().numpy()
        if not np.isfinite(v).all():
            raise ValueError("Nonfinite encoder state")
        out[i : i + len(v)] = v
        if i % (64 * batch_size) == 0:
            progress = {
                "completed": i + len(v),
                "total": len(texts),
                "seconds": time.monotonic() - start,
            }
            write_json(output / "encoding_progress.json", progress)
            print(json.dumps(progress), flush=True)
    del model
    if device == "mps":
        torch.mps.empty_cache()
    return out, {
        "model": name,
        "revision": revision,
        "prefix": prefix,
        "pool": pool,
        "dtype": "float32",
        "device": device,
        "batch_size": batch_size,
        "fixed_padding": 512,
        "max_observed_tokens": max(lengths),
        "truncated": 0,
        "seconds": time.monotonic() - start,
    }


def main():
    """Discovery only by default; exact same author/comment ledger for every stage."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--stage", choices=("base", "bge_large", "e5_large"), default="base")
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--group-plan-dir", type=Path)
    a = p.parse_args()
    bundles = json.loads(a.views.read_text(encoding="utf-8"))
    if not bundles or any(b["split"] != "discovery" for b in bundles):
        raise ValueError("This discovery runner must not open calibration/holdout")
    for b in bundles:
        validate_multicomment_bundle(b, units=12)
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a new empty output directory")
    start = time.monotonic()
    archive_source(ROOT, out / "source_snapshot.zip")
    rows = [c for b in bundles for v in b["views"] for c in v]
    texts = [r["body"] for r in rows]
    refs = np.array([b["discovery_role"] == "reference" for b in bundles])
    if not refs.any() or refs.all():
        raise ValueError("Need disjoint reference and comparison authors")
    manifest = {
        "protocol": "GRAIN_VM02_MULTICOMMENT_V1",
        "source_view_sha256": material_hash(bundles),
        "stage": a.stage,
        "authors": [b["author_id"] for b in bundles],
        "reference_mask": refs.tolist(),
        "labels_used": False,
        "code_sha256": code_fingerprint(ROOT),
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    write_json(out / "run_status.json", {"status": "RUNNING"})
    features, statuses, provenance = {}, {}, {}
    if a.stage == "base":
        vectorizers, style_scaler = {}, None
        for layout in ("chronological", "interleaved"):
            documents = []
            style = []
            for b in bundles:
                comments = [c for v in b["views"] for c in v]
                for group in view_layout(b, layout):
                    documents.append("\n\n".join(comments[i]["body"] for i in group))
                    units = np.asarray(
                        [style_features(comments[i]["body"]) for i in group]
                    )
                    style.append(np.r_[units.mean(0), units.std(0)])
            for analyzer, ngram in (("word", (1, 2)), ("char", (2, 5))):
                if analyzer not in vectorizers:
                    vectorizers[analyzer] = TfidfVectorizer(
                        analyzer=analyzer,
                        ngram_range=ngram,
                        max_features=20000,
                        min_df=2,
                        sublinear_tf=True,
                    ).fit(np.array(documents)[np.repeat(refs, 3)])
                vectorizer = vectorizers[analyzer]
                key = f"tfidf_{analyzer}__{layout}"
                features[key] = (
                    vectorizer.transform(documents)
                    .toarray()
                    .astype(np.float32)
                    .reshape(len(bundles), 3, -1)
                )
                joblib.dump(vectorizer, out / f"{key}.joblib")
            style = np.asarray(style)
            if style_scaler is None:
                style_scaler = StandardScaler().fit(style[np.repeat(refs, 3)])
            features[f"format_mean_std__{layout}"] = style_scaler.transform(
                style
            ).reshape(len(bundles), 3, -1)
        gf, statuses, provenance["grain"], uv = grain_scores(
            texts, bundles, np.repeat(refs, 3), ROOT
        )
        features.update(gf)
        provenance["reference_fit"] = (
            "48 chronological reference authors only; identical transforms for interleaved views"
        )
        np.savez_compressed(out / "unit_vectors.npz", grain_units=uv)
    else:
        from transformers import AutoTokenizer

        length_functions = []
        imported_groups = None
        if a.group_plan_dir:
            parent = json.loads(
                (a.group_plan_dir / "manifest.json").read_text(encoding="utf-8")
            )
            if parent["source_view_sha256"] != material_hash(bundles):
                raise ValueError("Grouping parent uses different comments")
            imported_groups = json.loads(
                (a.group_plan_dir / "group_ledger.json").read_text(encoding="utf-8")
            )
            if len(imported_groups) != len(bundles) * 3:
                raise ValueError("Grouping parent has wrong view count")
            provenance["group_plan_parent"] = parent["identity"]
        else:
            for name, rev, prefix, _ in MODELS.values():
                tokenizer = AutoTokenizer.from_pretrained(
                    name, revision=rev, local_files_only=True
                )
                length_functions.append(
                    lambda t, tok=tokenizer, pre=prefix: len(tok.encode(pre + t))
                )
        groups_by_view, joint_texts = [], []
        for b in bundles:
            for v in b["views"]:
                vt = [c["body"] for c in v]
                groups = (
                    imported_groups[len(groups_by_view)]
                    if imported_groups is not None
                    else common_groups(vt, length_functions)
                )
                if sorted(i for g in groups for i in g) != list(range(12)):
                    raise ValueError("Group plan must partition every view exactly")
                groups_by_view.append(groups)
                joint_texts.extend(
                    "\n\n".join(vt[i] for i in group) for group in groups
                )
        write_json(out / "group_ledger.json", groups_by_view)
        values, provenance[a.stage] = encode_progress(
            texts + joint_texts, MODELS[a.stage], a.device, out, batch_size=a.batch_size
        )
        units = values[: len(texts)].reshape(len(bundles), 36, -1)
        for layout in ("chronological", "interleaved"):
            features[f"{a.stage}_independent_mean__{layout}"] = np.asarray(
                [
                    [units[i, group].mean(0) for group in view_layout(b, layout)]
                    for i, b in enumerate(bundles)
                ]
            )
        at, joint = len(texts), []
        for groups in groups_by_view:
            joint.append(pool_groups(values[at : at + len(groups)], groups, 12))
            at += len(groups)
        features[f"{a.stage}_joint_local__chronological"] = np.asarray(joint).reshape(
            len(bundles), 3, -1
        )
        np.savez_compressed(out / "unit_vectors.npz", units=units)
        provenance["joint_scope"] = (
            "Shared complete-record groups, <=3 adjacent comments and <=512 tokens for both encoders; pooled by record count"
        )
    np.savez_compressed(out / "representations.npz", **features)
    np.savez_compressed(out / "support_status.npz", **statuses)
    write_json(out / "provenance.json", provenance)
    write_json(
        out / "run_status.json",
        {
            "status": "COMPLETE",
            "seconds": time.monotonic() - start,
            "methods": len(features),
            "comments": len(texts),
        },
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "stage": a.stage,
                "seconds": time.monotonic() - start,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
