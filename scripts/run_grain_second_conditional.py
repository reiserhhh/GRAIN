#!/usr/bin/env python3
"""S17: continue a fixed conditional-cumulant readout, reusing the S16 pilot."""

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
from grain_research.vector_mechanisms.direct_traits import tokenize_prompt
from grain_research.vector_mechanisms.prefix import capture_prefix, rotate_query
from grain_research.vector_mechanisms.readout_gain import multihead_gain
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.analyze_grain_author_discovery import scalar_metrics
from scripts.analyze_grain_query_representation import target_quality
from scripts.analyze_grain_teacher_student_bridge import paired_intervals
from scripts.run_grain_conditional_moments import verify as verify_s15
from scripts.run_grain_multiquery_sensitivity import token_groups
from scripts.run_grain_readout_gain import verify as verify_s16
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write
from scripts.run_grain_source_remainder import CONNECTION

ARMS = ("exact", "exact_mass_cov", "exact_mass_second")


def compact_packet(packet, reused):
    """Keep the three fixed scores, requiring complete finite author/view units."""
    views = packet["views"]
    if len(views) != 3 or {v["view"] for v in views} != {0, 1, 2}:
        raise ValueError("Three distinct views required")
    rows = []
    for view in sorted(views, key=lambda v: v["view"]):
        scores = {arm: float(view["scores"][arm]) for arm in ARMS}
        if not np.isfinite(list(scores.values())).all() or not (
            0 <= view["replay_error"] <= 1e-8
        ):
            raise ValueError("Finite replayed scores required")
        rows.append(
            {
                "view": view["view"],
                "scores": scores,
                "replay_error": view["replay_error"],
            }
        )
    return {
        "author_id": packet["author_id"],
        "fold": packet["fold"],
        "reused_s16": reused,
        "views": rows,
    }


def prepare(args):
    """Freeze all source hashes and reuse72 pilot prefixes without outcome access."""
    pilot, out = args.pilot.resolve(), args.output_dir
    _m, source, _s14, _s12, initial = verify_s16(pilot, args.device)
    _m, _s14, _s12, material = verify_s15(source, args.device)
    state = read(pilot / "run_status.json")
    if state["status"] != "COMPLETE" or state["calls"] != 72 or len(material) != 213:
        raise ValueError("Complete24-author pilot and213-author source required")
    if initial != material[:24]:
        raise ValueError("Pilot must be the declared source prefix")
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_packets").mkdir()
    files = [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/readout_gain.py",
        ROOT / "docs/GRAIN_SECOND_CONDITIONAL_S17_PLAN.md",
    ]
    manifest = {
        "protocol": "GRAIN_S17_SECOND_CONDITIONAL_V1",
        "pilot": str(pilot),
        "pilot_manifest_sha256": sha(pilot / "manifest.json"),
        "pilot_summary_sha256": sha(pilot / "summary.json"),
        "source": str(source),
        "source_manifest_sha256": sha(source / "manifest.json"),
        "source_packets": {
            f"author_{i:03d}.json": sha(
                source / "private_packets" / f"author_{i:03d}.json"
            )
            for i in range(213)
        },
        "pilot_packets": {
            f"author_{i:03d}.json": sha(
                pilot / "private_packets" / f"author_{i:03d}.json"
            )
            for i in range(24)
        },
        "author_membership_hash": material_hash([a["author_id"] for a in material]),
        "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
        "device": args.device,
        "authors": 213,
        "cached_authors": 24,
        "cached_prefixes": 72,
        "max_new_prefix_calls": 567,
        "max_seconds": 900,
        "arms": ARMS,
        "outcome_fit": False,
    }
    write(out / "manifest.json", manifest)
    for i in range(24):
        name = f"author_{i:03d}.json"
        write(
            out / "private_packets" / name,
            compact_packet(read(pilot / "private_packets" / name), True),
        )
    print(
        {
            "status": "PREPARED",
            "reused_authors": 24,
            "new_authors": 189,
            "new_prefix_calls": 567,
        },
        flush=True,
    )


def verify(out, device):
    """Revalidate both inherited lineages and every reused score packet."""
    m = read(out / "manifest.json")
    pilot, source = Path(m["pilot"]), Path(m["source"])
    if (
        m["device"] != device
        or sha(pilot / "manifest.json") != m["pilot_manifest_sha256"]
        or sha(pilot / "summary.json") != m["pilot_summary_sha256"]
        or sha(source / "manifest.json") != m["source_manifest_sha256"]
        or any(sha(ROOT / p) != h for p, h in m["code_hashes"].items())
        or any(
            sha(source / "private_packets" / p) != h
            for p, h in m["source_packets"].items()
        )
        or any(
            sha(pilot / "private_packets" / p) != h
            for p, h in m["pilot_packets"].items()
        )
    ):
        raise ValueError("Frozen S17 inputs changed")
    _m, inherited, _s14, _s12, _initial = verify_s16(pilot, device)
    _m, s14, s12, material = verify_s15(source, device)
    if (
        inherited != source
        or material_hash([a["author_id"] for a in material])
        != m["author_membership_hash"]
    ):
        raise ValueError("Source cohort changed")
    for name in m["pilot_packets"]:
        if read(out / "private_packets" / name) != compact_packet(
            read(pilot / "private_packets" / name), True
        ):
            raise ValueError("Cached pilot score changed")
    return m, source, s14, s12, material


def run(args):
    """Measure only the189 remaining authors with a bounded frozen prefix reader."""
    out = args.output_dir
    manifest, source, s14, s12, material = verify(out, args.device)
    previous = (
        read(out / "run_status.json")
        if (out / "run_status.json").exists()
        else {"new_calls": 0, "seconds": 0}
    )
    with np.load(s14 / "private_references.npz", allow_pickle=False) as z:
        references = {name: z[name].astype(np.float32) for name in z.files}
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device=args.device, dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    start = time.monotonic()

    def status(name):
        state = {
            "status": name,
            "new_calls": previous["new_calls"] + observer.calls,
            "seconds": previous["seconds"] + time.monotonic() - start,
            "authors": len(list((out / "private_packets").glob("author_*.json"))),
            "cached_prefixes": 72,
            "labels_read": False,
        }
        write(out / "run_status.json", state)
        print(state, flush=True)

    try:
        for i, author in enumerate(material):
            path = out / "private_packets" / f"author_{i:03d}.json"
            if path.exists():
                packet = read(path)
                compact_packet(packet, i < 24)
                if packet["author_id"] != author["author_id"] or packet[
                    "reused_s16"
                ] != (i < 24):
                    raise ValueError("Resume author/cache mismatch")
                continue
            if (
                previous["seconds"] + time.monotonic() - start
                >= manifest["max_seconds"]
                or previous["new_calls"] + observer.calls + 3
                > manifest["max_new_prefix_calls"]
            ):
                status("PAUSED_RESOURCE_BOUNDARY")
                return
            old = read(source / "private_packets" / f"author_{i:03d}.json")
            fold = old["fold"]
            with np.load(
                s12
                / "analysis/private_models"
                / f"kv_pairing_pca64_configuration_response_fold{fold}.npz",
                allow_pickle=False,
            ) as z:
                beta, intercept = z["coef"], float(z["intercept"])
            rows = []
            for view, payload in enumerate(author["views"]):
                ids, user, spans = tokenize_prompt(
                    payload["texts"], "neuroticism", observer.tokenizer
                )
                if material_hash(ids) != payload["prompt_sha256"]:
                    raise ValueError("Frozen prompt changed")
                groups = token_groups(observer, user, spans, ids)
                cache, trace = capture_prefix(observer, ids)
                query = rotate_query(
                    references[str(fold)],
                    cache["last_cos"],
                    cache["last_sin"],
                    cache["scaling"],
                )
                vectors, _detail = multihead_gain(cache, groups, query, beta)
                scores = {arm: float(vectors[arm] @ beta + intercept) for arm in ARMS}
                error = max(
                    abs(scores[a] - old["views"][view]["scores"][b])
                    for a, b in (
                        ("exact", "exact"),
                        ("exact_mass_cov", "exact_mass_moment"),
                    )
                )
                if trace["forbidden_calls"]:
                    raise ValueError("Prefix crossed frozen boundary")
                rows.append({"view": view, "scores": scores, "replay_error": error})
            write(
                path,
                compact_packet(
                    {"author_id": author["author_id"], "fold": fold, "views": rows},
                    False,
                ),
            )
            if (i + 1) % 16 == 0:
                status("RUNNING")
        status("COMPLETE")
    except BaseException as exc:
        status("FAILED")
        write(out / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def fidelity(frame, cohort):
    """Report uncalibrated internal approximation quality separately by cohort."""
    variance = float(np.var(frame.exact))
    if not variance > 0:
        raise ValueError("Nonconstant exact reference required")
    rows = []
    for arm in ARMS:
        error = frame[arm] - frame.exact
        rows.append(
            {
                "cohort": cohort,
                "n": len(frame),
                "arm": arm,
                "exact_r": scalar_metrics(frame.exact, frame[arm])["pearson_r"],
                "mse_over_exact_variance": float(np.mean(error**2)) / variance,
                "mean_error": float(error.mean()),
                "centered_error_variance": float(np.var(error)),
            }
        )
    return rows


def evaluate(args):
    """Freeze all scores before connecting already-exposed source outcomes once."""
    out = args.output_dir
    _m, _source, s14, _s12, material = verify(out, args.device)
    state = read(out / "run_status.json")
    if (
        state["status"] != "COMPLETE"
        or state["new_calls"] != 567
        or state["authors"] != 213
    ):
        raise ValueError("Complete fixed continuation required")
    if (out / "scores_frozen.json").exists():
        raise ValueError("Preserve frozen evaluation")
    rows, maximum, hashes = [], 0.0, {}
    for i, author in enumerate(material):
        path = out / "private_packets" / f"author_{i:03d}.json"
        packet = compact_packet(read(path), i < 24)
        if packet["author_id"] != author["author_id"]:
            raise ValueError("Evaluation author mismatch")
        rows.append(
            {
                "author_id": author["author_id"],
                "reused_s16": i < 24,
                **{
                    arm: float(np.mean([v["scores"][arm] for v in packet["views"]]))
                    for arm in ARMS
                },
            }
        )
        maximum = max(maximum, *(v["replay_error"] for v in packet["views"]))
        hashes[path.name] = sha(path)
    frame = pd.DataFrame(rows)
    if not frame.author_id.is_unique:
        raise ValueError("Unique author units required")
    internal = fidelity(frame[~frame.reused_s16], "new189") + fidelity(frame, "all213")
    pd.DataFrame(internal).to_csv(out / "internal_fidelity.csv", index=False)
    frozen = out / "private_scores_before_outcomes.csv"
    frame.to_csv(frozen, index=False)
    write(
        out / "scores_frozen.json",
        {"sha256": sha(frozen), "packet_hashes": hashes, "labels_used": False},
    )
    inputs = pd.read_csv(s14 / "private_inputs.csv", dtype={"author_id": str})
    labels = pd.read_csv(
        CONNECTION / "private_predictions.csv",
        dtype={"author_id": str},
        usecols=["author_id", "query", "y_true"],
    )
    labels = labels[labels["query"] == "neuroticism"][["author_id", "y_true"]]
    frame = frame.merge(
        inputs[["author_id", "teacher", "null_prediction"]],
        on="author_id",
        validate="one_to_one",
    ).merge(labels, on="author_id", validate="one_to_one")
    if len(frame) != 213 or not np.isfinite(frame.y_true).all():
        raise ValueError("Complete source outcome coverage required")
    metrics, pairs = [], []
    for arm in ARMS:
        external = scalar_metrics(frame.y_true, frame[arm])
        metrics.append(
            {
                "arm": arm,
                **target_quality(frame.teacher, frame[arm], frame.null_prediction),
                "source_r": external["pearson_r"],
                "source_p": external["pearson_p"],
                "source_r_lower": external["r_lower"],
                "source_r_upper": external["r_upper"],
            }
        )
        pairs.append(
            {
                "arm": arm,
                **paired_intervals(frame.y_true, frame[arm], frame.exact, False),
            }
        )
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(pairs).to_csv(out / "source_paired_intervals.csv", index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "n": 213,
            "new_authors": 189,
            "new_prefix_calls": state["new_calls"],
            "cached_prefixes": 72,
            "seconds": state["seconds"],
            "replay_error": maximum,
            "scores_sha256": sha(frozen),
            "internal_fidelity": internal,
            "metrics": metrics,
            "source_paired_intervals": pairs,
            "human_outcome_fit": False,
        },
    )
    print(pd.DataFrame(internal).to_string(index=False), flush=True)
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)


def main():
    """Expose preparation, bounded collection and one frozen source evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run", "evaluate"])
    parser.add_argument("--pilot", type=Path, default=BASE / "readout_gain_s16_r1")
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "second_conditional_s17_r1"
    )
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "evaluate": evaluate}[args.mode](args)


if __name__ == "__main__":
    main()
