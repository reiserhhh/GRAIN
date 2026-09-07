#!/usr/bin/env python3
"""D2: query-guided real evidence removal and intermediate-vector interventions."""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.direct_traits import (
    SYSTEM,
    candidates,
    native_readout,
    render,
)
from grain_research.vector_mechanisms.external_material import unpack_features
from grain_research.vector_mechanisms.trace import DecoderObserver, Patch
from scripts.run_grain_vector_mechanism_external_features import (
    bundle_arms,
    collect_encoder,
)

MODEL = (
    ROOT
    / "results/grain_vector_mechanisms/local_observers/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)
BASE = ROOT / "results/grain_vector_mechanisms"
ANCHORS = {
    "TF_cont": (
        "Making decisions through impersonal logic, analytical consistency and objective principles.",
        "Making decisions through personal values, empathy and concern for people's feelings.",
    ),
    "neuroticism": (
        "Remaining calm, emotionally steady and relaxed when facing everyday difficulties.",
        "Experiencing worry, distress, emotional vulnerability and negative feelings in everyday life.",
    ),
}


def read(path):
    """Read a private local artifact without printing its contents."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    """Persist replayable strict JSON."""
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def sha(path):
    """Hash local source/input bytes."""
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def key(value, salt="d2_author_20260908"):
    """Label-free reproducible ordering."""
    return hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()


def choose_comments(vectors, poles, lengths, k=4):
    """Select absolute trait-contrast relevance and a disjoint length-matched control."""
    vectors, poles = np.asarray(vectors, float), np.asarray(poles, float)
    lengths = np.asarray(lengths, int)
    if (
        vectors.ndim != 2
        or poles.shape != (2, vectors.shape[1])
        or lengths.shape != (len(vectors),)
        or len(vectors) < 2 * k
        or not np.isfinite(vectors).all()
        or not np.isfinite(poles).all()
        or np.any(lengths < 1)
    ):
        raise ValueError("Invalid paired comment/pole vectors")
    v = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    q = poles / np.maximum(np.linalg.norm(poles, axis=1, keepdims=True), 1e-12)
    signed = v @ (q[1] - q[0])
    selected = sorted(np.argsort(-np.abs(signed), kind="stable")[:k].tolist())
    rest = [i for i in range(len(vectors)) if i not in selected]
    target = lengths[selected].sum()
    control = min(
        itertools.combinations(rest, k),
        key=lambda ids: (
            abs(int(lengths[list(ids)].sum() - target)),
            key(ids, "length_tie"),
        ),
    )
    return selected, list(control), signed


def variants(texts, selected, control):
    """Keep original order in the five fixed natural evidence conditions."""
    if (
        len(set(selected)) != 4
        or len(set(control)) != 4
        or set(selected) & set(control)
    ):
        raise ValueError("Require disjoint four-comment conditions")
    return {
        "full": texts,
        "remove_selected": [t for i, t in enumerate(texts) if i not in selected],
        "remove_control": [t for i, t in enumerate(texts) if i not in control],
        "selected_only": [texts[i] for i in selected],
        "control_only": [texts[i] for i in control],
    }


def select_membership(registry, excluded, counts, holdout_only=False):
    """Select source-only authors without overlap or outcome-dependent ordering."""
    ids = registry["author_ids"]
    chosen = []
    for task, group, number in counts:
        eligible = [
            i
            for i, uid in enumerate(ids)
            if registry["all_arms_ready_reference_free"][i]
            and registry["source_group"][i] == group
            and uid not in excluded
        ]
        ranked = sorted(eligible, key=lambda i: key(ids[i]))[:number]
        if number < 2 or len(ranked) != number or (number % 2 and not holdout_only):
            raise ValueError("Insufficient source-ready authors or invalid split count")
        for rank, i in enumerate(ranked):
            chosen.append(
                {
                    "author_id": str(ids[i]),
                    "task": task,
                    "source_group": group,
                    "split": "held_author" if holdout_only or rank % 2 else "discovery",
                    "profile_index": i,
                }
            )
    return chosen


def prepare(args):
    """Freeze new source-author material, independent profiles and query selection."""
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    excluded = set(
        read(BASE / "d1_author_discovery_r1/membership_snapshot.json")[
            "private_author_ids"
        ]
    )
    extra_exclusions = {}
    for path in args.exclude_membership:
        excluded.update(row["author_id"] for row in read(path))
        extra_exclusions[str(path)] = sha(path)
    registry = np.load(
        BASE / "vm05_same_ready_registry_r1/private_ready_registry.npz",
        allow_pickle=False,
    )
    profiles = np.load(
        BASE / "vm05_profile_material_r1/private_profile_features.npz",
        allow_pickle=False,
    )
    ids = registry["author_ids"]
    if not np.array_equal(ids, profiles["author_ids"]):
        raise ValueError("Profile and author registry mismatch")
    counts = (
        ("TF_cont", "mbti_only", args.tf_authors),
        ("neuroticism", "big5_only", args.n_authors),
    )
    chosen = select_membership(registry, excluded, counts, args.holdout_only)
    # Commit membership before model scoring; this stage does not read any labels.
    write(out / "membership.json", chosen)
    material_path = BASE / "vm05_qualification_r2/common_views.json"
    source = read(material_path)
    wanted = {r["author_id"] for r in chosen}
    bundles = {b["author_id"]: b for b in source if b["author_id"] in wanted}
    del source
    if set(bundles) != wanted:
        raise ValueError("Missing complete author material")
    torch.set_num_threads(2)
    e5_path = args.e5_dir
    tokenizer = AutoTokenizer.from_pretrained(
        e5_path, local_files_only=True, trust_remote_code=False
    )
    encoder = (
        AutoModel.from_pretrained(
            e5_path, local_files_only=True, trust_remote_code=False, dtype=torch.float32
        )
        .to(args.device)
        .eval()
    )
    anchor_texts = [p for poles in ANCHORS.values() for p in poles]
    q = collect_encoder(
        anchor_texts, tokenizer, encoder, "query: ", "mean", device=args.device
    )
    del encoder, tokenizer
    gc.collect()
    if args.device == "mps":
        torch.mps.empty_cache()
    qt = AutoTokenizer.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False
    )
    cache = {}
    for shard in sorted((BASE / "vm05_full_vectors_r1/e5_large").glob("shard_*")):
        sm = read(shard / "manifest.json")
        for uid in sm["authors"]:
            if uid in wanted:
                cache[uid] = shard / "private_cache" / (uid + ".npz")
    rows, inputs, grain, landmarks, semantics = [], {}, [], [], []
    for person in chosen:
        uid = person["author_id"]
        texts = bundle_arms(bundles[uid])["cue_mask"]
        path = cache[uid]
        metadata = read(path.with_suffix(".json"))
        if sha(path) != metadata["feature_sha256"]:
            raise ValueError("Changed cached comment vectors")
        with np.load(path, allow_pickle=False) as z:
            arrays = unpack_features(dict(z), metadata["feature_aliases"])
        vectors = arrays["cue_mask_comment_vectors"]
        if vectors.shape[:2] != (3, 12) or len(texts) != 36:
            raise ValueError("Complete three-view author needed")
        inputs[str(path.relative_to(ROOT))] = sha(path)
        gi = person["profile_index"]
        grain.append(profiles["cue_mask_whitened"][gi])
        landmarks.append(profiles["cue_mask_canonical_distances"][gi])
        semantics.append(vectors.mean(axis=(0, 1)))
        poles = q[0:2] if person["task"] == "TF_cont" else q[2:4]
        for view in range(2):
            body = texts[12 * view : 12 * (view + 1)]
            lengths = [len(qt.encode(t, add_special_tokens=False)) for t in body]
            selected, control, scores = choose_comments(vectors[view], poles, lengths)
            rows.append(
                {
                    **person,
                    "view": view,
                    "texts": body,
                    "selected": selected,
                    "control": control,
                    "selection_scores": scores.tolist(),
                    "comment_lengths": lengths,
                }
            )
    np.savez_compressed(
        out / "private_profiles.npz",
        author_ids=np.asarray([r["author_id"] for r in chosen]),
        grain_whitened=np.asarray(grain),
        canonical_distances=np.asarray(landmarks),
        e5_mean=np.asarray(semantics),
        query_poles=q,
    )
    write(out / "private_material.json", rows)
    manifest = {
        "protocol": "GRAIN_D2_REAL_AUTHOR_PATH_V1",
        "authors": len(chosen),
        "author_views": len(rows),
        "arm": "cue_mask",
        "site": "block_16",
        "device": args.device,
        "dtype": "float32",
        "anchors": ANCHORS,
        "material_sha256": sha(out / "private_material.json"),
        "profiles_sha256": sha(out / "private_profiles.npz"),
        "common_views_sha256": sha(material_path),
        "encoder_cache_sha256": inputs,
        "d1_excluded": True,
        "extra_excluded_membership_sha256": extra_exclusions,
        "holdout_only": args.holdout_only,
        "labels_read": False,
        "bridge_outcomes_read": False,
        "model_dir": str(MODEL),
        "model_sha256": sha(MODEL / "model.safetensors"),
        "script_sha256": sha(__file__),
        "max_calls": len(rows) * 9 + 4,
        "max_seconds": args.max_seconds,
    }
    write(out / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "MATERIAL_READY",
                "authors": len(chosen),
                "views": len(rows),
                "expected_calls": manifest["max_calls"],
            }
        ),
        flush=True,
    )


def capture(observer, ids, alphabet, patch=None):
    """Capture two query-end sites and only the final vocabulary logits."""
    pos = len(ids) - 1
    device = next(observer.model.parameters()).device
    tokens = torch.tensor([ids], device=device)
    with observer.hooks([pos], patch) as states, torch.inference_mode():
        output = observer.model(
            input_ids=tokens,
            attention_mask=torch.ones_like(tokens),
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )
    observer.calls += 1
    observer.input_tokens += len(ids)
    logits = output.logits[0, -1].detach().float().cpu().numpy()
    result = native_readout(logits, alphabet)
    if not np.isfinite(logits).all() or not all(
        np.isfinite(v).all() for v in states.values()
    ):
        raise ValueError("Nonfinite observation")
    return states, result


def run(args):
    """Run complete author units and preserve progress without starting old queues."""
    out = args.output_dir
    manifest = read(out / "manifest.json")
    if sha(out / "private_material.json") != manifest["material_sha256"]:
        raise ValueError("Material changed after selection")
    if (
        sha(__file__) != manifest["script_sha256"]
        or sha(MODEL / "model.safetensors") != manifest["model_sha256"]
    ):
        raise ValueError("Frozen execution changed")
    units = read(out / "private_material.json")
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device=args.device, dtype="float32", max_tokens=3072
    )
    observer.sites = {
        "block_16": observer.model.model.layers[16],
        "final_norm": observer.model.model.norm,
    }
    write(out / "observer.json", observer.manifest())
    packets = out / "private_packets"
    packets.mkdir(exist_ok=True)
    start = time.monotonic()
    completed, checks = [], []

    def status(state):
        payload = {
            "status": state,
            "completed_author_views": len(completed),
            "planned_author_views": len(units),
            "calls_this_invocation": observer.calls,
            "seconds": time.monotonic() - start,
            "labels_read": False,
            "bridge_outcomes_read": False,
        }
        write(out / "run_status.json", payload)
        print(json.dumps(payload), flush=True)

    try:
        for row in units:
            case = f"{row['author_id']}_{row['view']}"
            target = packets / (case + ".json")
            if target.exists():
                saved = read(target)
                if (
                    saved["state_sha256"] != sha(packets / (case + ".npz"))
                    or len(saved["readouts"]) != 9
                ):
                    raise ValueError("Incomplete saved author unit")
                completed.append(case)
                continue
            if row["view"] == 0 and time.monotonic() - start > manifest["max_seconds"]:
                status("PAUSED_TIME_AT_AUTHOR_BOUNDARY")
                return
            alphabet = candidates(row["task"], observer.tokenizer)
            natural, vectors, ids_by_condition = {}, {}, {}
            for condition, texts in variants(
                row["texts"], row["selected"], row["control"]
            ).items():
                user, _ = render(texts, row["task"])
                ids = observer.prompt_ids(SYSTEM, user)
                states, result = capture(observer, ids, alphabet)
                natural[condition] = {**result, "tokens": len(ids)}
                ids_by_condition[condition] = ids
                for site, h in states.items():
                    vectors[f"{condition}__{site}"] = h[0]
            ids = ids_by_condition["full"]
            for donor in ("remove_selected", "remove_control"):
                for alpha in (0.5, 1.0):
                    name = f"patch_{donor}_{alpha}"
                    patch = Patch(
                        "block_16",
                        len(ids) - 1,
                        vectors[f"{donor}__block_16"],
                        alpha=alpha,
                    )
                    states, result = capture(observer, ids, alphabet, patch)
                    natural[name] = {**result, "tokens": len(ids)}
                    for site, h in states.items():
                        vectors[f"{name}__{site}"] = h[0]
            if not any(c["task"] == row["task"] for c in checks):
                _, repeat = capture(observer, ids, alphabet)
                _, self_patch = capture(
                    observer,
                    ids,
                    alphabet,
                    Patch("block_16", len(ids) - 1, vectors["full__block_16"]),
                )
                delta = max(
                    abs(
                        repeat["conditional_score"]
                        - natural["full"]["conditional_score"]
                    ),
                    abs(
                        self_patch["conditional_score"]
                        - natural["full"]["conditional_score"]
                    ),
                )
                checks.append(
                    {"task": row["task"], "repeat_self_max_score_error": delta}
                )
                write(out / "instrument_checks.json", checks)
                if delta > 1e-5:
                    raise ValueError("Repeat/self-patch changes the observation")
            np.savez_compressed(packets / (case + ".npz"), **vectors)
            write(
                target,
                {k: row[k] for k in ("author_id", "task", "view", "split")}
                | {"readouts": natural, "state_sha256": sha(packets / (case + ".npz"))},
            )
            completed.append(case)
            if observer.calls > manifest["max_calls"]:
                raise RuntimeError("Call budget exceeded")
            if row["view"] == 1:
                status("RUNNING")
        status("COMPLETE")
    except BaseException as exc:
        status("INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED")
        write(out / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def main():
    """Prepare or execute the fixed real-author packet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "run"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument(
        "--e5-dir",
        type=Path,
        default=Path.home()
        / ".cache/huggingface/hub/models--intfloat--e5-large-v2/snapshots/f169b11e22de13617baa190a028a32f3493550b6",
    )
    parser.add_argument("--tf-authors", type=int, default=40)
    parser.add_argument("--n-authors", type=int, default=24)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--exclude-membership", type=Path, action="append", default=[])
    parser.add_argument("--holdout-only", action="store_true")
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to(ROOT / "results"):
        raise ValueError("Private material must remain in ignored results/")
    (prepare if args.phase == "prepare" else run)(args)


if __name__ == "__main__":
    main()
