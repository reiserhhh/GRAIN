#!/usr/bin/env python3
"""S1: same-author common edits, differentiable readouts and joint-edit responses."""

from __future__ import annotations

import argparse
import gc
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
    question,
    render,
)
from grain_research.vector_mechanisms.external_material import unpack_features
from grain_research.vector_mechanisms.trace import DecoderObserver, Patch
from scripts.run_grain_real_author_path import (
    ANCHORS,
    BASE,
    MODEL,
    capture,
    read,
    select_membership,
    sha,
    write,
)
from scripts.run_grain_vector_mechanism_external_features import (
    bundle_arms,
    collect_encoder,
)

QUERIES = ("TF_cont", "neuroticism", "confirmation_seeking")
POLES = {
    **ANCHORS,
    "confirmation_seeking": (
        "Making everyday judgments independently without asking others for reassurance or approval.",
        "Seeking other people's reassurance, confirmation or approval when making everyday judgments.",
    ),
}
NARROW_QUESTION = (
    "Estimate the writer's tendency to seek other people's reassurance, confirmation or approval "
    "when making everyday judgments. Use 0 for very low, 9 for very high, and the digits in "
    "between for intermediate tendencies. Return only one digit from 0 to 9."
)


def unit_vectors(x):
    """Normalize finite rows for the frozen E5 anchor contrast."""
    x = np.asarray(x, float)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite fixed-encoder vectors")
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def common_edits(vectors, poles, lengths, tolerance=0.2):
    """Fix the union of query top evidence and disjoint low-relevance matched controls."""
    vectors, poles = unit_vectors(vectors), unit_vectors(poles)
    lengths = np.asarray(lengths, int)
    if vectors.ndim != 2 or poles.shape != (3, 2, vectors.shape[1]):
        raise ValueError("Three query-pole pairs and comment vectors required")
    if lengths.shape != (len(vectors),) or np.any(lengths < 1) or len(vectors) < 6:
        raise ValueError("Positive lengths and enough disjoint evidence required")
    signed = vectors @ (poles[:, 1] - poles[:, 0]).T
    e1 = sorted(set(np.argmax(np.abs(signed), axis=0).tolist()))
    target = int(lengths[e1].sum())
    relevance = np.max(np.abs(signed), axis=1)
    choices = list(
        itertools.combinations([j for j in range(len(vectors)) if j not in e1], len(e1))
    )

    def cost(indices):
        gap = abs(int(lengths[list(indices)].sum()) - target) / target
        return (
            gap > tolerance,
            float(relevance[list(indices)].mean()) if gap <= tolerance else gap,
            gap if gap <= tolerance else float(relevance[list(indices)].mean()),
            indices,
        )

    e2 = list(min(choices, key=cost))
    gap = abs(int(lengths[e2].sum()) - target) / target
    return (
        e1,
        e2,
        signed,
        {"relative_length_gap": gap, "within_tolerance": gap <= tolerance},
    )


def edit_indices(count, e1, e2):
    """Remove the same disjoint evidence sets for every query, preserving order."""
    if not e1 or len(e1) != len(e2) or set(e1) & set(e2):
        raise ValueError("Disjoint same-count edits required")
    if len(set(e1)) != len(e1) or len(set(e2)) != len(e2):
        raise ValueError("Duplicate edit index")
    if min(e1 + e2) < 0 or max(e1 + e2) >= count:
        raise ValueError("Edit out of bounds")
    return {
        name: [j for j in range(count) if j not in removed]
        for name, removed in (
            ("full", set()),
            ("e1", set(e1)),
            ("e2", set(e2)),
            ("joint", set(e1) | set(e2)),
        )
    }


def query_prompt(texts, query):
    """Use unchanged native TF/N wording and an explicitly separate narrow query."""
    base_task = "neuroticism" if query == "confirmation_seeking" else query
    user, spans = render(texts, base_task)
    if query == "confirmation_seeking":
        user = user.replace(question(base_task), NARROW_QUESTION)
    return user, spans


def native_alphabet(query, tokenizer):
    """Reuse verified single-token alphabets without adding outcome calibration."""
    return candidates(
        "neuroticism" if query == "confirmation_seeking" else query, tokenizer
    )


def token_groups(observer, user, spans, ids):
    """Map only excerpt-body tokens to comments in the actual chat-template sequence."""
    text = observer.tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if text.count(user) != 1:
        raise ValueError("Ambiguous rendered user span")
    shift = text.index(user)
    encoded = observer.tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True
    )
    if list(encoded["input_ids"]) != ids:
        raise ValueError("Offset tokenization disagrees with actual template")
    groups = []
    for lo, hi in spans:
        group = [
            j
            for j, (a, b) in enumerate(encoded["offset_mapping"])
            if b > a >= lo + shift and b <= hi + shift
        ]
        if not group:
            raise ValueError("Comment has no complete token positions")
        groups.append(group)
    return groups


def conditional_torch(logits, alphabet):
    """Differentiable version of the exact native candidate-conditioned score."""
    indices = torch.as_tensor(alphabet["token_ids"], device=logits.device)
    values = torch.as_tensor(
        alphabet["values"], dtype=logits.dtype, device=logits.device
    )
    return torch.softmax(logits[indices], dim=0) @ values


def differentiable_capture(observer, ids, alphabet, groups):
    """Differentiate the frozen suffix from a block16 leaf, not model parameters.

    Prefix activations do not retain a graph, keeping the24GB Mac within budget.
    Comment attribution is block16 gradient-times-state, not input-embedding IG.
    """
    saved = {}

    def boundary(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        leaf = h.detach().requires_grad_(True)
        saved["leaf"] = leaf
        return (leaf, *output[1:]) if isinstance(output, tuple) else leaf

    device = next(observer.model.parameters()).device
    tokens = torch.tensor([ids], device=device)
    handle = observer.model.model.layers[16].register_forward_hook(boundary)
    try:
        with torch.enable_grad():
            output = observer.model(
                input_ids=tokens,
                attention_mask=torch.ones_like(tokens),
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
            logits = output.logits[0, -1]
            score = conditional_torch(logits, alphabet)
            (gradient,) = torch.autograd.grad(score, saved["leaf"])
        observer.calls += 1
        observer.input_tokens += len(ids)
        leaf = saved["leaf"].detach()
        products = (gradient * leaf).sum(-1)[0]
        result = native_readout(logits.detach().float().cpu().numpy(), alphabet)
        attribution = [float(products[g].sum().cpu()) for g in groups]
        h = leaf[0, -1].float().cpu().numpy().copy()
        g = gradient[0, -1].detach().float().cpu().numpy().copy()
        if (
            not np.isfinite(h).all()
            or not np.isfinite(g).all()
            or not np.isfinite(attribution).all()
        ):
            raise ValueError("Nonfinite state/gradient/attribution")
        result["torch_score_error"] = abs(
            float(score.detach().cpu()) - result["conditional_score"]
        )
        return h, g, attribution, result
    finally:
        handle.remove()


def prepare(args):
    """Freeze new source-only authors, common edits and cached E5 comment vectors."""
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    excluded = set(
        read(BASE / "d1_author_discovery_r1/membership_snapshot.json")[
            "private_author_ids"
        ]
    )
    sources = [
        BASE / name / "membership.json"
        for name in ("d2_real_author_path_r1", "d3b_fresh_author_path_r1")
    ]
    for source in sources:
        excluded.update(row["author_id"] for row in read(source))
    with np.load(
        BASE / "vm05_same_ready_registry_r1/private_ready_registry.npz",
        allow_pickle=False,
    ) as registry:
        chosen = select_membership(
            registry,
            excluded,
            (("TF_cont", "mbti_only", 12), ("neuroticism", "big5_only", 12)),
            True,
        )
    # Alternate source groups so the first8 pilot authors include4 from each.
    chosen = [
        person for pair in zip(chosen[:12], chosen[12:], strict=True) for person in pair
    ]
    for index, person in enumerate(chosen):
        person["source_task"] = person.pop("task")
        person["panel_index"] = index
    write(out / "membership.json", chosen)
    wanted = {r["author_id"] for r in chosen}
    material_path = BASE / "vm05_qualification_r2/common_views.json"
    bundles = {
        b["author_id"]: b for b in read(material_path) if b["author_id"] in wanted
    }
    if set(bundles) != wanted:
        raise ValueError("Missing selected author text")
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(
        args.e5_dir, local_files_only=True, trust_remote_code=False
    )
    encoder = (
        AutoModel.from_pretrained(
            args.e5_dir,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
        )
        .to(args.device)
        .eval()
    )
    poles = collect_encoder(
        [s for pair in POLES.values() for s in pair],
        tokenizer,
        encoder,
        "query: ",
        "mean",
        device=args.device,
    ).reshape(3, 2, -1)
    del encoder, tokenizer
    gc.collect()
    if args.device == "mps":
        torch.mps.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False
    )
    cache, hashes = {}, {}
    for shard in sorted((BASE / "vm05_full_vectors_r1/e5_large").glob("shard_*")):
        for uid in read(shard / "manifest.json")["authors"]:
            if uid in wanted:
                cache[uid] = shard / "private_cache" / (uid + ".npz")
    rows, encoded = [], []
    for person in chosen:
        uid = person["author_id"]
        path = cache[uid]
        metadata = read(path.with_suffix(".json"))
        if sha(path) != metadata["feature_sha256"]:
            raise ValueError("Changed frozen E5 cache")
        hashes[str(path.relative_to(ROOT))] = sha(path)
        with np.load(path, allow_pickle=False) as z:
            vectors = unpack_features(dict(z), metadata["feature_aliases"])[
                "cue_mask_comment_vectors"
            ]
        texts = bundle_arms(bundles[uid])["cue_mask"]
        if vectors.shape[:2] != (3, 12) or len(texts) != 36:
            raise ValueError("Complete three-view inputs required")
        for view in range(2):
            body = texts[12 * view : 12 * (view + 1)]
            lengths = [len(tokenizer.encode(t, add_special_tokens=False)) for t in body]
            e1, e2, signed, matching = common_edits(vectors[view], poles, lengths)
            rows.append(
                {
                    **person,
                    "view": view,
                    "texts": body,
                    "e1": e1,
                    "e2": e2,
                    "lengths": lengths,
                    "signed_relevance": signed.tolist(),
                    "matching": matching,
                    "joint": person["panel_index"] < 8 and view == 0,
                }
            )
            encoded.append(vectors[view])
    write(out / "private_material.json", rows)
    np.savez_compressed(
        out / "private_encoder.npz", comment_vectors=np.asarray(encoded), poles=poles
    )
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_S1_MULTIQUERY_SENSITIVITY_V1",
            "authors": 24,
            "views": 48,
            "queries": QUERIES,
            "anchors": POLES,
            "narrow_question": NARROW_QUESTION,
            "model_dir": str(MODEL),
            "model_sha256": sha(MODEL / "model.safetensors"),
            "material_sha256": sha(out / "private_material.json"),
            "encoder_sha256": sha(out / "private_encoder.npz"),
            "cache_hashes": hashes,
            "membership_sha256": sha(out / "membership.json"),
            "script_sha256": sha(__file__),
            "common_views_sha256": sha(material_path),
            "excluded_memberships": {str(p): sha(p) for p in sources},
            "dtype": "float32",
            "device": args.device,
            "site": "block_16",
            "candidate_score": "native conditional mean, same TF/N readouts",
            "gradient_scope": "block16 full-state leaf; frozen parameters; suffix graph only",
            "attribution": "block16 gradient times state; zero hidden reference; NOT input gradients",
            "input_gradients": "deferred for24GB memory budget; primary site gradient retained",
            "length_match_tolerance": 0.2,
            "max_forwards": 750,
            "max_backwards": 144,
            "max_seconds": args.max_seconds,
            "psychological_labels_read": False,
            "third_query_is_not_a_validated_attachment_scale": True,
        },
    )
    print(
        json.dumps(
            {
                "status": "MATERIAL_FROZEN",
                "authors": 24,
                "views": 48,
                "unmatched_views": sum(
                    not r["matching"]["within_tolerance"] for r in rows
                ),
            }
        ),
        flush=True,
    )


def run(args):
    """Acquire complete author units, resume safely, and account for cumulative cost."""
    out = args.output_dir
    manifest = read(out / "manifest.json")
    for name, digest in (
        ("private_material.json", "material_sha256"),
        ("private_encoder.npz", "encoder_sha256"),
        ("membership.json", "membership_sha256"),
    ):
        if sha(out / name) != manifest[digest]:
            raise ValueError("Frozen input changed")
    if (
        sha(__file__) != manifest["script_sha256"]
        or sha(MODEL / "model.safetensors") != manifest["model_sha256"]
    ):
        raise ValueError("Frozen script/model changed")
    if args.device != manifest["device"]:
        raise ValueError("Do not silently change the scoring device")
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device=args.device, dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    observer.sites = {"block_16": observer.model.model.layers[16]}
    write(out / "observer.json", observer.manifest())
    material = read(out / "private_material.json")
    packets = out / "private_packets"
    packets.mkdir(exist_ok=True)
    cost_file = out / "cost_ledger.json"
    ledger = (
        read(cost_file)
        if cost_file.exists()
        else {"forwards": 0, "backwards": 0, "seconds": 0.0}
    )
    prior = dict(ledger)
    checks_path = out / "instrument_checks.json"
    checks = read(checks_path) if checks_path.exists() else {}
    start, backwards = time.monotonic(), 0
    completed = set()

    def update(state):
        payload = {
            "status": state,
            "completed_author_views": len(completed),
            "completed_authors": len(completed) // 2,
            "planned_authors": 24,
            "forwards": prior["forwards"] + observer.calls,
            "backwards": prior["backwards"] + backwards,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "labels_read": False,
        }
        write(cost_file, {k: payload[k] for k in ("forwards", "backwards", "seconds")})
        write(out / "run_status.json", payload)
        print(json.dumps(payload), flush=True)

    try:
        for row_index, row in enumerate(material):
            stem = f"panel_{row['panel_index']:02d}_view_{row['view']}"
            target, arrays = packets / (stem + ".json"), packets / (stem + ".npz")
            if target.exists():
                old = read(target)
                if (
                    old["state_sha256"] != sha(arrays)
                    or old["author_id"] != row["author_id"]
                    or set(old["queries"]) != set(QUERIES)
                ):
                    raise ValueError("Invalid saved complete view")
                completed.add(stem)
                continue
            if row["view"] == 0:
                if row["panel_index"] >= args.max_authors:
                    update("PAUSED_PILOT_BOUNDARY")
                    return
                if (
                    prior["seconds"] + time.monotonic() - start
                    >= manifest["max_seconds"]
                ):
                    update("PAUSED_TIME_AT_AUTHOR_BOUNDARY")
                    return
            indices = edit_indices(12, row["e1"], row["e2"])
            vectors, results = {}, {}
            for query in QUERIES:
                alphabet = native_alphabet(query, observer.tokenizer)
                user, spans = query_prompt(row["texts"], query)
                ids = observer.prompt_ids(SYSTEM, user)
                groups = token_groups(observer, user, spans, ids)
                h, g, attribution, score = differentiable_capture(
                    observer, ids, alphabet, groups
                )
                backwards += 1
                vectors[f"{query}__full"], vectors[f"{query}__gradient"] = h, g
                scores = {"full": {**score, "tokens": len(ids)}}
                for edit in ("e1", "e2", "joint") if row["joint"] else ("e1", "e2"):
                    edited, _ = query_prompt(
                        [row["texts"][j] for j in indices[edit]], query
                    )
                    edit_ids = observer.prompt_ids(SYSTEM, edited)
                    state, score = capture(observer, edit_ids, alphabet)
                    vectors[f"{query}__{edit}"] = state["block_16"][0]
                    scores[edit] = {**score, "tokens": len(edit_ids)}
                for edit in ("e1", "e2"):
                    _, score = capture(
                        observer,
                        ids,
                        alphabet,
                        Patch("block_16", len(ids) - 1, vectors[f"{query}__{edit}"]),
                    )
                    scores["patch_" + edit] = {**score, "tokens": len(ids)}
                if query not in checks:
                    _, repeat = capture(observer, ids, alphabet)
                    _, self_patch = capture(
                        observer, ids, alphabet, Patch("block_16", len(ids) - 1, h)
                    )
                    error = max(
                        abs(
                            s["conditional_score"] - scores["full"]["conditional_score"]
                        )
                        for s in (repeat, self_patch)
                    )
                    checks[query] = {
                        "repeat_self_max_error": error,
                        "gradient_norm": float(np.linalg.norm(g)),
                        "torch_numpy_score_error": scores["full"]["torch_score_error"],
                    }
                    write(checks_path, checks)
                    if error > 1e-5 or np.linalg.norm(g) == 0:
                        raise ValueError(
                            "Differentiable readout failed instrument check"
                        )
                results[query] = {
                    "readouts": scores,
                    "comment_attribution": attribution,
                    "body_token_counts": [len(group) for group in groups],
                    "alphabet": alphabet,
                }
            np.savez_compressed(arrays, **vectors)
            write(
                target,
                {
                    "author_id": row["author_id"],
                    "source_group": row["source_group"],
                    "panel_index": row["panel_index"],
                    "view": row["view"],
                    "material_row": row_index,
                    "queries": results,
                    "state_sha256": sha(arrays),
                },
            )
            completed.add(stem)
            if (
                prior["forwards"] + observer.calls > manifest["max_forwards"]
                or prior["backwards"] + backwards > manifest["max_backwards"]
            ):
                raise RuntimeError("Frozen call budget exceeded")
            if row["view"] == 1:
                update("RUNNING")
            if args.device == "mps":
                torch.mps.empty_cache()
        update("COMPLETE")
    except BaseException as exc:
        update("INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED")
        write(out / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def main():
    """Prepare the fixed panel or acquire a bounded resumable subset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "run"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--max-authors", type=int, default=8)
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument(
        "--e5-dir",
        type=Path,
        default=Path.home()
        / ".cache/huggingface/hub/models--intfloat--e5-large-v2/snapshots/f169b11e22de13617baa190a028a32f3493550b6",
    )
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to(ROOT / "results"):
        raise ValueError("Private material must stay in ignored results/")
    if not 1 <= args.max_authors <= 24:
        raise ValueError("Author cap outside frozen panel")
    (prepare if args.phase == "prepare" else run)(args)


if __name__ == "__main__":
    main()
