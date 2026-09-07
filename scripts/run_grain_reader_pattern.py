#!/usr/bin/env python3
"""S18: frozen reader-pattern intervention at the actual layer17 head output."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.direct_traits import candidates, tokenize_prompt
from grain_research.vector_mechanisms.reader_pattern import (
    effect_decomposition,
    reader_patterns,
    split_reader_component,
)
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.analyze_grain_author_discovery import scalar_metrics
from scripts.analyze_grain_teacher_student_bridge import paired_intervals
from scripts.run_grain_attention_pairing import capture_head, patch_head_score
from scripts.run_grain_conditional_moments import verify as verify_s15
from scripts.run_grain_multiquery_sensitivity import token_groups
from scripts.run_grain_query_representation import pairing_vector, validate_packet
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write


def prepare(args):
    """Build forward patterns from training authors, without new response fitting."""
    source, out = args.source.resolve(), args.output_dir
    _m, s14, s12, material = verify_s15(source, args.device)
    inputs = pd.read_csv(s14 / "private_inputs.csv", dtype={"author_id": str})
    names = [a["author_id"] for a in material]
    if list(inputs.author_id) != names or len(names) != 213:
        raise ValueError("Complete ordered213-author source required")
    vectors, hashes = [], {}
    for i, author in enumerate(material):
        path = s12 / "private_packets" / f"author_{i:03d}.json"
        validate_packet(path, author)
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
            vectors.append(z["kv_pairing"].mean(0))
        hashes[str(path.relative_to(s12))] = sha(path)
        hashes[str(path.with_suffix(".npz").relative_to(s12))] = sha(
            path.with_suffix(".npz")
        )
    x, folds = np.asarray(vectors), inputs.fold.to_numpy()
    arrays, details = {}, {}
    for fold in range(5):
        model = (
            s12
            / "analysis/private_models"
            / f"kv_pairing_pca64_configuration_response_fold{fold}.npz"
        )
        hashes[str(model.relative_to(s12))] = sha(model)
        with np.load(model, allow_pickle=False) as z:
            beta = z["coef"]
        train = folds != fold
        patterns, detail = reader_patterns(x[train], beta)
        arrays[f"{fold}_beta"] = beta
        for name, value in patterns.items():
            arrays[f"{fold}_{name}"] = value
        details[str(fold)] = {
            **detail,
            "training_membership_sha256": material_hash(
                np.asarray(names)[train].tolist()
            ),
        }
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_packets").mkdir()
    np.savez_compressed(out / "private_patterns.npz", **arrays)
    write(out / "private_pattern_details.json", details)
    files = [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/reader_pattern.py",
        ROOT / "docs/GRAIN_READER_PATTERN_S18_PLAN.md",
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_S18_READER_PATTERN_V1",
            "source": str(source),
            "source_manifest_sha256": sha(source / "manifest.json"),
            "source_s12_hashes": hashes,
            "patterns_sha256": sha(out / "private_patterns.npz"),
            "pattern_details_sha256": sha(out / "private_pattern_details.json"),
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "device": args.device,
            "author_membership_sha256": material_hash(names[:24]),
            "authors": 24,
            "views_per_author": 1,
            "max_forwards": 168,
            "max_seconds": 900,
            "outcome_fit": False,
            "questionnaire_access": False,
        },
    )
    print({"status": "PREPARED", "authors": 24, "forwards": 168}, flush=True)


def verify(out, device):
    """Revalidate fixed patterns, code, readers and underlying feature provenance."""
    m = read(out / "manifest.json")
    source = Path(m["source"])
    if (
        m["device"] != device
        or sha(source / "manifest.json") != m["source_manifest_sha256"]
        or sha(out / "private_patterns.npz") != m["patterns_sha256"]
        or sha(out / "private_pattern_details.json") != m["pattern_details_sha256"]
        or any(sha(ROOT / p) != h for p, h in m["code_hashes"].items())
    ):
        raise ValueError("S18 frozen inputs changed")
    _m, s14, s12, material = verify_s15(source, device)
    if (
        any(sha(s12 / p) != h for p, h in m["source_s12_hashes"].items())
        or material_hash([a["author_id"] for a in material[:24]])
        != m["author_membership_sha256"]
    ):
        raise ValueError("S18 source geometry or panel changed")
    return m, s14, s12, material[:24]


def run(args):
    """Execute seven declared full-score interventions per fixed source author."""
    out = args.output_dir
    manifest, s14, s12, material = verify(out, args.device)
    previous = (
        read(out / "run_status.json")
        if (out / "run_status.json").exists()
        else {"calls": 0, "seconds": 0}
    )
    inputs = pd.read_csv(
        s14 / "private_inputs.csv", dtype={"author_id": str}
    ).set_index("author_id")
    with np.load(out / "private_patterns.npz", allow_pickle=False) as z:
        saved = {name: z[name] for name in z.files}
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device=args.device, dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    alphabet = candidates("neuroticism", observer.tokenizer)
    start = time.monotonic()

    def status(name):
        state = {
            "status": name,
            "calls": previous["calls"] + observer.calls,
            "seconds": previous["seconds"] + time.monotonic() - start,
            "authors": len(list((out / "private_packets").glob("author_*.json"))),
            "labels_read": False,
        }
        write(out / "run_status.json", state)
        print(state, flush=True)

    try:
        for i, author in enumerate(material):
            path = out / "private_packets" / f"author_{i:03d}.json"
            if path.exists():
                if read(path)["author_id"] != author["author_id"]:
                    raise ValueError("Resume author mismatch")
                continue
            if (
                previous["seconds"] + time.monotonic() - start
                >= manifest["max_seconds"]
                or previous["calls"] + observer.calls + 7 > manifest["max_forwards"]
            ):
                status("PAUSED_RESOURCE_BOUNDARY")
                return
            payload, fold = (
                author["views"][0],
                int(inputs.loc[author["author_id"], "fold"]),
            )
            beta = saved[f"{fold}_beta"]
            ids, user, spans = tokenize_prompt(
                payload["texts"], "neuroticism", observer.tokenizer
            )
            if material_hash(ids) != payload["prompt_sha256"]:
                raise ValueError("Frozen prompt changed")
            groups = token_groups(observer, user, spans, ids)
            native, cache = capture_head(observer, ids, alphabet)
            d, head_error = pairing_vector(cache, groups)
            old_path = s12 / "private_packets" / f"author_{i:03d}.json"
            old = read(old_path)
            with np.load(old_path.with_suffix(".npz"), allow_pickle=False) as z:
                feature_error = float(np.max(np.abs(d - z["kv_pairing"][0])))
            native_error = abs(native - old["views"][0]["score"])
            h = cache["native_head"].reshape(-1)
            self_score = patch_head_score(observer, ids, alphabet, h)
            self_error = abs(native - self_score)
            if feature_error > 1e-8 or native_error > 2e-5 or self_error > 2e-5:
                raise ValueError("Source feature/native/self replay failed")
            removed = patch_head_score(observer, ids, alphabet, h - d)
            effects = []
            for geometry in ("euclidean", "covariance"):
                selected, null = split_reader_component(
                    d, beta, saved[f"{fold}_{geometry}"]
                )
                selected_score = patch_head_score(observer, ids, alphabet, h - selected)
                null_score = patch_head_score(observer, ids, alphabet, h - null)
                effects.append(
                    {
                        "geometry": geometry,
                        **effect_decomposition(
                            native, removed, selected_score, null_score
                        ),
                        "selected_score": selected_score,
                        "null_score": null_score,
                        "selected_norm_over_pairing": float(
                            np.linalg.norm(selected) / np.linalg.norm(d)
                        ),
                        "null_norm_over_pairing": float(
                            np.linalg.norm(null) / np.linalg.norm(d)
                        ),
                        "null_reader_score": float(beta @ null),
                        "selected_reader_error": float(beta @ selected - beta @ d),
                    }
                )
            write(
                path,
                {
                    "author_id": author["author_id"],
                    "fold": fold,
                    "view": 0,
                    "native": native,
                    "full_removed": removed,
                    "reader_component": float(beta @ d),
                    "native_error": native_error,
                    "feature_error": feature_error,
                    "self_error": self_error,
                    "head_error": head_error,
                    "pairing_norm": float(np.linalg.norm(d)),
                    "head_norm": float(np.linalg.norm(h)),
                    "effects": effects,
                },
            )
            if (i + 1) % 4 == 0:
                status("RUNNING")
        status("COMPLETE")
    except BaseException as exc:
        status("FAILED")
        write(out / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def evaluate(args):
    """Summarize complete intervention effects and their nonlinear interaction."""
    out = args.output_dir
    _m, _s14, _s12, material = verify(out, args.device)
    state = read(out / "run_status.json")
    if state["status"] != "COMPLETE" or state["calls"] != 168:
        raise ValueError("Complete fixed24-author intervention required")
    if (out / "summary.json").exists():
        raise ValueError("Preserve completed analysis")
    rows, errors = (
        [],
        {
            name: 0.0
            for name in ("native_error", "feature_error", "self_error", "head_error")
        },
    )
    for i, author in enumerate(material):
        packet = read(out / "private_packets" / f"author_{i:03d}.json")
        if packet["author_id"] != author["author_id"] or len(packet["effects"]) != 2:
            raise ValueError("Complete matched author units required")
        for name, value in errors.items():
            errors[name] = max(value, packet[name])
        for effect in packet["effects"]:
            rows.append(
                {
                    "author_id": author["author_id"],
                    "reader_component": packet["reader_component"],
                    **effect,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "private_effects.csv", index=False)
    metrics = []
    for geometry, group in frame.groupby("geometry", sort=False):
        full, selected, null = [
            group[name].to_numpy()
            for name in ("full_effect", "selected_effect", "null_effect")
        ]
        energy = float(np.mean(full**2))
        metrics.append(
            {
                "geometry": geometry,
                "n": len(group),
                "full_effect_rms": float(np.sqrt(energy)),
                "selected_effect_rms": float(np.sqrt(np.mean(selected**2))),
                "null_effect_rms": float(np.sqrt(np.mean(null**2))),
                "selected_full_r": scalar_metrics(full, selected)["pearson_r"],
                "null_full_r": scalar_metrics(full, null)["pearson_r"],
                "selected_mse_over_full_energy": float(
                    np.mean((selected - full) ** 2) / energy
                ),
                "null_mse_over_full_energy": float(
                    np.mean((null - full) ** 2) / energy
                ),
                "interaction_rms_over_full": float(
                    np.sqrt(np.mean(group.interaction**2) / energy)
                ),
                "median_selected_norm_over_pairing": float(
                    group.selected_norm_over_pairing.median()
                ),
                "maximum_selected_norm_over_pairing": float(
                    group.selected_norm_over_pairing.max()
                ),
                "reader_full_effect_r": scalar_metrics(full, group.reader_component)[
                    "pearson_r"
                ],
                "max_null_reader_score": float(group.null_reader_score.abs().max()),
                "selected_vs_null_r_interval": paired_intervals(
                    full, selected, null, False
                ),
            }
        )
    pd.DataFrame(
        [{k: v for k, v in m.items() if not isinstance(v, dict)} for m in metrics]
    ).to_csv(out / "metrics.csv", index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "n": 24,
            "forwards": state["calls"],
            "seconds": state["seconds"],
            "labels_read": False,
            "metrics": metrics,
            "replay_errors": errors,
            "effects_sha256": sha(out / "private_effects.csv"),
        },
    )
    print(
        pd.DataFrame(
            [{k: v for k, v in m.items() if not isinstance(v, dict)} for m in metrics]
        ).to_string(index=False),
        flush=True,
    )


def main():
    """Prepare, collect or analyze a fixed no-questionnaire intervention panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run", "evaluate"])
    parser.add_argument(
        "--source", type=Path, default=BASE / "conditional_moments_s15_r1"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "reader_pattern_s18_r1"
    )
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "evaluate": evaluate}[args.mode](args)


if __name__ == "__main__":
    main()
