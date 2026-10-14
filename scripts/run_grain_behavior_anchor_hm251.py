#!/usr/bin/env python3
"""Observe independent public behavior anchors before complete-text reader actions."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_behavior_anchor as M
from grain_research.construction import math_selector_matrix as S
from grain_research.vector_mechanisms import direct_traits as D
from grain_research.vector_mechanisms.navigation import MODEL
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_prospective_evidence_hm218 as H

CONFIG = "configs/grain_behavior_anchor_hm251.json"
DEFAULT = ROOT / "results/grain_math_response_v1/behavior_anchor_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_BEHAVIOR_ANCHOR_PLAN_HM251.md",
    "grain_research/construction/math_behavior_anchor.py",
    "scripts/run_grain_behavior_anchor_hm251.py",
    "tests/test_grain_behavior_anchor.py",
    "grain_research/construction/math_balanced_evidence.py",
    "grain_research/construction/math_selector_matrix.py",
    "grain_research/vector_mechanisms/direct_traits.py",
    "grain_research/vector_mechanisms/trace.py",
    "scripts/run_grain_observed_family_hm170.py",
    "scripts/run_grain_prospective_evidence_hm218.py",
)


def local_environment():
    """Use only the already-pinned local checkpoint, with no remote code."""
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )


def source_ids(body, retained, tokenizer):
    """Concatenate fixed token blocks so event-at-slot balance also fixes token bags."""
    ids = tokenizer.encode(
        "The following excerpts were written by one anonymous writer.",
        add_special_tokens=False,
    )
    for slot, event in enumerate(retained):
        ids += tokenizer.encode(f"\n\nExcerpt {slot + 1}:\n", add_special_tokens=False)
        ids += tokenizer.encode(body[event], add_special_tokens=False)
    ids += tokenizer.encode("\nEvidence ends here.", add_special_tokens=False)
    return ids


def prepare(out):
    """Fix item keys, sixteen real-input families and all tokens before any observation."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("No overwrite/restart")
    cfg = R.read(ROOT / CONFIG)
    if (
        cfg["panel"] != list(range(6, 12)) + list(range(18, 24))
        or cfg["source_target"] != 3
    ):
        raise ValueError("Changed finite panel/criterion")
    parent = ROOT / cfg["parent"]
    old = R.check(
        parent, ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    meta = H.selected(
        parent / "observations.npz", ("conditions", "lengths", "panel_indices")
    )
    units = R.read(parent / "units.json")
    local_environment()
    tok = AutoTokenizer.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False
    )
    rows = []
    for target in cfg["targets"]:
        if sum(map(len, cfg["items"][target].values())) != 10:
            raise ValueError("Changed public ten-item key")
        for pole in ("high", "low"):
            for original in cfg["items"][target][pole]:
                sentence = "I " + original[0].lower() + original[1:]
                rows.append(
                    {
                        "target": target,
                        "pole": pole,
                        "original": original,
                        "sentence": sentence,
                        "input_ids": tok.encode(sentence, add_special_tokens=False),
                        "kind": "item",
                    }
                )
    bits, removed, slots = M.fixed_family()
    readers, references, token_range = [], [], []
    for u, old_u in enumerate(cfg["panel"]):
        body = S.extract_bodies(
            tok.decode(units[old_u * 15]["input_ids"], skip_special_tokens=False)
        )
        if [len(tok.encode(t, add_special_tokens=False)) for t in body] != meta[
            "lengths"
        ][old_u].tolist():
            raise ValueError("Changed historical public event tokens")
        for q, target in enumerate(D.BIG5):
            if (
                D.tokenize_prompt(body, target, tok, cfg["max_tokens"])[0]
                != units[old_u * 15 + q]["input_ids"]
            ):
                raise ValueError("Changed historical rendering")
        denominator = len(source_ids(body, list(range(12)), tok))
        references.append(denominator)
        for k, removal in enumerate(removed):
            kept = [i for i in range(12) if i not in removal]
            source = source_ids(body, kept, tok)
            native = [
                D.tokenize_prompt(
                    [body[i] for i in kept], target, tok, cfg["max_tokens"]
                )[0]
                for target in cfg["reader_targets"]
            ]
            if len(source) + 16 > cfg["max_tokens"]:
                raise ValueError("SOURCE input too long; no truncation")
            rows.append(
                {
                    "kind": "text",
                    "author": u,
                    "old_panel_author": old_u,
                    "candidate": k,
                    "denominator": denominator,
                    "input_ids": source,
                }
            )
            readers.append(
                {
                    "author": u,
                    "candidate": k,
                    "old_panel_author": old_u,
                    "input_ids": native,
                }
            )
            token_range += [len(source), *map(len, native)]
    if len(rows) != cfg["max_source_rows"] or len(readers) != cfg["max_native_batches"]:
        raise ValueError("Changed fifty-item/twelve-author full scope")
    model_files = {
        p.name: R.digest(p)
        for p in MODEL.iterdir()
        if p.is_file() and not p.name.startswith("._")
    }
    if model_files != old["model_files"]:
        raise ValueError("Changed frozen observer")
    inputs = [
        parent / p
        for p in (
            "manifest.json",
            "prepared_seal.json",
            "collected_seal.json",
            "analyzed_seal.json",
            "verification_seal.json",
            "observations.npz",
            "units.json",
            "head.npz",
        )
    ]
    (out / "source_batches").mkdir(parents=True, exist_ok=False)
    (out / "native_batches").mkdir(exist_ok=False)
    R.write(out / "source_rows.json", rows)
    R.write(out / "reader_rows.json", readers)
    R.archive(
        out / "observations.npz",
        conditions=meta["conditions"][cfg["panel"]],
        panel_indices=meta["panel_indices"][cfg["panel"]],
        bits=bits,
        removals=removed,
        slots=slots,
        reference_lengths=np.asarray(references),
    )
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "model_files": model_files,
            "alphabet": old["alphabet"],
            "original_scope_complete": False,
            "source_uses_author_labels_or_reader_answers": False,
        },
    )
    quality = {
        "authors": 12,
        "items": 50,
        "source_rows": len(rows),
        "native_batches": len(readers),
        "native_inquiries": 2 * len(readers),
        "input_tokens_range": [min(token_range), max(token_range)],
        "reference_lengths": references,
        "historical_cue_mask_inherited": True,
        "new_public_authors_or_untouched_lockbox": False,
        "new_API_private_train": 0,
    }
    R.write(out / "input_quality.json", quality)
    R.seal(
        out,
        "prepared_seal.json",
        (
            "source_rows.json",
            "reader_rows.json",
            "observations.npz",
            "input_quality.json",
        ),
    )
    H.status(
        out,
        {
            "status": "PREPARED",
            "completed_source_rows": 0,
            "completed_native_batches": 0,
            "model_seconds": 0,
        },
    )
    print(json.dumps(quality), flush=True)


def observe(out, stage):
    """Observe each fixed row once with one shared nonrenewable model-time budget."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    seals = () if stage == "source" else ("source_seal.json", "design_seal.json")
    manifest = R.check(out, seals)
    cfg = manifest["config"]
    expected = "PREPARED" if stage == "source" else "DESIGNED"
    if R.read(out / "runtime.json")["status"] != expected:
        raise ValueError("No retry of started/terminal stage")
    for name, value in manifest["model_files"].items():
        if R.digest(MODEL / name) != value:
            raise ValueError("Changed model identity")
    rows = R.read(
        out / ("source_rows.json" if stage == "source" else "reader_rows.json")
    )
    elapsed = (
        0.0
        if stage == "source"
        else R.read(out / "source_runtime.json")["model_seconds"]
    )
    completed = 0
    lock = out / "ACTIVE.lock"
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    try:
        local_environment()
        torch.set_num_threads(2)
        observer = DecoderObserver.load(
            str(MODEL),
            device=cfg["device"],
            dtype=cfg["dtype"],
            max_tokens=cfg["max_tokens"],
        )
        alphabet = manifest["alphabet"]["token_ids"]
        if stage == "native":
            head = (
                observer.model.lm_head.weight.detach()[alphabet].float().cpu().numpy()
            )
            if not np.array_equal(
                head,
                H.selected(ROOT / cfg["parent"] / "head.npz", ("weights",))["weights"],
            ):
                raise ValueError("Changed native head")
            R.archive(out / "head.npz", weights=head)
        for index, row in enumerate(rows):
            if elapsed >= cfg["max_model_seconds"]:
                raise RuntimeError("Shared finite cap reached; no renewal/retry")
            H.status(
                out,
                {
                    "status": stage.upper() + "_RUNNING",
                    "pid": os.getpid(),
                    "pending_row": index,
                    "completed_rows": completed,
                    "model_seconds": elapsed,
                },
            )
            sequences = [row["input_ids"]] if stage == "source" else row["input_ids"]
            pad = observer.tokenizer.pad_token_id
            if pad is None:
                pad = observer.tokenizer.eos_token_id
            ids = torch.full(
                (len(sequences), max(map(len, sequences))),
                pad,
                dtype=torch.long,
                device=cfg["device"],
            )
            mask = torch.zeros_like(ids)
            for j, sequence in enumerate(sequences):
                ids[j, : len(sequence)] = torch.tensor(sequence, device=cfg["device"])
                mask[j, : len(sequence)] = 1
            stamp = time.monotonic()
            try:
                with torch.inference_mode():
                    output = observer.model.model(
                        input_ids=ids,
                        attention_mask=mask,
                        use_cache=False,
                        return_dict=True,
                    )
                    if stage == "source":
                        state = (
                            output.last_hidden_state[0, : len(sequences[0])]
                            .float()
                            .cpu()
                            .numpy()
                            .astype(float)
                        )
                        embedding = (
                            observer.model.model.embed_tokens(ids)[
                                0, : len(sequences[0])
                            ]
                            .float()
                            .cpu()
                            .numpy()
                            .astype(float)
                        )
                        denominator = (
                            len(sequences[0])
                            if row["kind"] == "item"
                            else row["denominator"]
                        )
                        if (
                            not np.isfinite(state).all()
                            or not np.isfinite(embedding).all()
                        ):
                            raise ValueError("Nonfinite SOURCE observation")
                        R.archive(
                            out / "source_batches" / f"{index:03d}.npz",
                            vector=state.sum(0) / denominator,
                            embedding=embedding.sum(0) / denominator,
                            tokens=np.array(len(sequences[0])),
                            denominator=np.array(denominator),
                        )
                    else:
                        last = torch.tensor(
                            [len(s) - 1 for s in sequences], device=cfg["device"]
                        )
                        states = output.last_hidden_state[
                            torch.arange(2, device=cfg["device"]), last
                        ]
                        logits = (
                            observer.model.lm_head(states)
                            .float()
                            .cpu()
                            .numpy()
                            .astype(float)
                        )
                        if not np.isfinite(logits).all():
                            raise ValueError("Nonfinite native response")
                        R.archive(
                            out / "native_batches" / f"{index:03d}.npz",
                            states=states.float().cpu().numpy(),
                            logits=logits[:, alphabet],
                            normalizer=logsumexp(logits, -1),
                        )
            finally:
                elapsed += time.monotonic() - stamp
            completed += 1
            H.status(
                out,
                {
                    "status": stage.upper() + "_RUNNING",
                    "pid": os.getpid(),
                    "pending_row": None,
                    "completed_rows": completed,
                    "model_seconds": elapsed,
                },
            )
            if completed % 24 == 0:
                print(
                    f"HM251 {stage} {completed}/{len(rows)}; cumulative_model_seconds={elapsed:.3f}",
                    flush=True,
                )
        status = {
            "status": stage.upper() + "_COLLECTED",
            "pid": None,
            "pending_row": None,
            "completed_rows": completed,
            "model_seconds": elapsed,
            "new_API_private_train": 0,
        }
        R.write(out / f"{stage}_runtime.json", status)
        H.status(out, status)
        files = tuple(f"{stage}_batches/{i:03d}.npz" for i in range(len(rows))) + (
            f"{stage}_runtime.json",
        )
        if stage == "native":
            files += ("head.npz",)
        R.seal(out, f"{stage}_seal.json", files)
        print(json.dumps(status), flush=True)
    except BaseException as exc:
        H.status(
            out,
            {
                "status": "FAILED",
                "stage": stage,
                "completed_rows": completed,
                "model_seconds": elapsed,
                "error": repr(exc),
                "no_auto_retry": True,
            },
        )
        raise
    finally:
        lock.unlink(missing_ok=True)


def source_arrays(out):
    """Load complete query-free item/text states, never native outcomes."""
    rows = R.read(out / "source_rows.json")
    vectors, embeddings = [], []
    for i in range(len(rows)):
        z = H.selected(out / "source_batches" / f"{i:03d}.npz", ("vector", "embedding"))
        vectors.append(z["vector"])
        embeddings.append(z["embedding"])
    return np.asarray(vectors), np.asarray(embeddings), rows


def design(out):
    """Seal behavior meanings and positive comparison laws BEFORE native judgments."""
    manifest = R.check(out, ("source_seal.json",))
    cfg, z = manifest["config"], R.prepared(out)
    if R.read(out / "runtime.json")["status"] != "SOURCE_COLLECTED":
        raise ValueError("Complete source observation required before design")
    h, embedding, rows = source_arrays(out)
    directions = M.public_directions(h[:50], rows[:50], cfg["targets"])
    text = h[50:].reshape(12, 16, -1)
    scores = text @ directions.T
    weights, accounts, descriptors = [], [], []
    for u in range(12):
        a = M.balanced_design(scores[u], z["conditions"][u])
        weights.append(a["weights"])
        accounts.append(M.walsh_accounts(scores[u], a["weights"])["by_order"])
        descriptors.append(
            {
                "author": u,
                "old_panel_author": cfg["panel"][u],
                "source_action": a["source_action"].tolist(),
                "balance_error": a["balance_error"],
                "dual_gap": a["dual_gap"],
                "count_distribution": a["count_distribution"].tolist(),
                "support_size": int(np.count_nonzero(np.any(a["weights"] > 0, axis=0))),
                "source_direction_prediction": "positive C reader action; no reorientation after outcomes",
            }
        )
    weights = np.asarray(weights)
    additive = embedding[50:].reshape(12, 16, -1)
    embedding_action = np.einsum("uk,ukd->ud", weights[:, 0] - weights[:, 1], additive)
    quality = {
        "source_C_actions": [r["source_action"][3] for r in descriptors],
        "max_balance_error": max(r["balance_error"] for r in descriptors),
        "max_embedding_bag_action": float(np.max(abs(embedding_action))),
        "support_sizes": [r["support_size"] for r in descriptors],
        "anchor_cosines": (directions @ directions.T).tolist(),
        "native_outcomes_read": False,
        "no_orthogonalization_or_outcome_fit": True,
    }
    if quality["max_embedding_bag_action"] > 1e-8:
        raise ValueError(
            "SOURCE additive token bag not balanced; no model response stage"
        )
    R.archive(
        out / "design.npz",
        directions=directions,
        scores=scores,
        weights=weights,
        source_order_accounts=np.asarray(accounts),
        embedding_actions=embedding_action,
    )
    R.write(out / "designs.json", descriptors)
    R.write(out / "design_quality.json", quality)
    R.seal(
        out, "design_seal.json", ("design.npz", "designs.json", "design_quality.json")
    )
    H.status(
        out,
        {
            "status": "DESIGNED",
            "model_seconds": R.read(out / "source_runtime.json")["model_seconds"],
            "completed_source_rows": 242,
            "completed_native_batches": 0,
        },
    )
    print(json.dumps(quality), flush=True)


def native_arrays(out):
    """Load all native inputs; no sparse-support/positive-author filtering."""
    arrays = [
        H.selected(out / "native_batches" / f"{i:03d}.npz", ("logits", "states"))
        for i in range(192)
    ]
    return np.asarray([r["logits"] for r in arrays]).reshape(12, 16, 2, 10), np.asarray(
        [r["states"] for r in arrays]
    ).reshape(12, 16, 2, -1)


def analyze(out):
    """Retain every signed source-to-reader action and interaction order."""
    manifest = R.check(
        out, ("source_seal.json", "design_seal.json", "native_seal.json")
    )
    cfg = manifest["config"]
    z = H.selected(out / "design.npz", ("scores", "weights", "source_order_accounts"))
    logits, _ = native_arrays(out)
    y = logsumexp(logits[..., 5:], -1) - logsumexp(logits[..., :5], -1)
    action = np.einsum("uk,ukq->uq", z["weights"][:, 0] - z["weights"][:, 1], y)
    orders = np.asarray(
        [M.walsh_accounts(y[u], z["weights"][u])["by_order"] for u in range(12)]
    )
    ci = S.bootstrap(action, cfg["bootstrap_draws"], cfg["bootstrap_seed"])
    surplus = action[:, 1] - action[:, 0]
    source = np.einsum(
        "uk,ukq->uq", z["weights"][:, 0] - z["weights"][:, 1], z["scores"]
    )
    summary = {
        "authors": 12,
        "mean_A_C": action.mean(0).tolist(),
        "CI_A_C": ci.tolist(),
        "positive_authors_A_C": (action > 0).sum(0).tolist(),
        "RMS_A_C": np.sqrt(np.mean(action**2, 0)).tolist(),
        "bank_means_A_C": [a.mean(0).tolist() for a in np.split(action, 2)],
        "C_minus_A_mean": float(surplus.mean()),
        "C_minus_A_CI": S.bootstrap(
            surplus, cfg["bootstrap_draws"], cfg["bootstrap_seed"]
        ).tolist(),
        "source_C_mean": float(source[:, 3].mean()),
        "source_C_positive_authors": int(np.sum(source[:, 3] > 1e-8)),
        "source_mean_ENACO": source.mean(0).tolist(),
        "source_order_means_ENACO": z["source_order_accounts"].mean(0).tolist(),
        "native_order_means_A_C": orders.mean(0).tolist(),
        "model_seconds": R.read(out / "native_runtime.json")["model_seconds"],
        "all192_native_batches_complete": True,
        "human_personality_accuracy": False,
        "new_factor_or_original_scope_complete": False,
    }
    rows = [
        {
            "author": u,
            "old_panel_author": cfg["panel"][u],
            "source_C": float(source[u, 3]),
            "native_A": float(action[u, 0]),
            "native_C": float(action[u, 1]),
            "C_minus_A": float(surplus[u]),
        }
        for u in range(12)
    ]
    R.table(out / "author_actions.csv", rows)
    R.archive(
        out / "analysis.npz",
        native_actions=action,
        native_y=y,
        native_orders=orders,
        source_actions=source,
    )
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analyzed_seal.json",
        ("author_actions.csv", "analysis.npz", "summary.json"),
    )
    print(json.dumps(summary), flush=True)


def verify(out):
    """Use literal sums for the finite scientific accounts; no precision search."""
    R.check(
        out,
        (
            "source_seal.json",
            "design_seal.json",
            "native_seal.json",
            "analyzed_seal.json",
        ),
    )
    p = R.prepared(out)
    z = H.selected(out / "design.npz", ("directions", "scores", "weights"))
    a = H.selected(
        out / "analysis.npz",
        ("native_actions", "native_y", "native_orders", "source_actions"),
    )
    h, e, _ = source_arrays(out)
    logits, states = native_arrays(out)
    head = H.selected(out / "head.npz", ("weights",))["weights"]
    source_error, action_error, balance_error, walsh_error = 0.0, 0.0, 0.0, 0.0
    for u in range(12):
        nu = z["weights"][u, 0] - z["weights"][u, 1]
        kept = np.array([[j for j in range(12) if j not in s] for s in p["removals"]])
        counts = p["conditions"][u, kept].sum(1)
        controls = np.column_stack(
            (
                p["slots"].reshape(16, -1),
                z["scores"][u][:, [0, 1, 2, 4]],
                counts[:, None] == np.arange(9),
            )
        )
        for col in controls.T:
            balance_error = max(
                balance_error, abs(sum(float(nu[k] * col[k]) for k in range(16)))
            )
        for q in range(5):
            literal = np.array(
                [
                    sum(
                        float(v * d)
                        for v, d in zip(
                            h[50 + u * 16 + k], z["directions"][q], strict=True
                        )
                    )
                    for k in range(16)
                ]
            )
            source_error = max(
                source_error, float(np.max(abs(literal - z["scores"][u, :, q])))
            )
            source_error = max(
                source_error,
                abs(
                    sum(float(nu[k] * literal[k]) for k in range(16))
                    - a["source_actions"][u, q]
                ),
            )
        for q in range(2):
            yy = []
            for k in range(16):
                x = logits[u, k, q]
                high, low = max(x[5:]), max(x[:5])
                odds = (
                    high
                    + np.log(sum(np.exp(x[5:] - high)))
                    - low
                    - np.log(sum(np.exp(x[:5] - low)))
                )
                yy.append(float(odds))
                action_error = max(action_error, abs(odds - a["native_y"][u, k, q]))
            literal = sum(float(nu[k] * yy[k]) for k in range(16))
            action_error = max(action_error, abs(literal - a["native_actions"][u, q]))
            walsh_error = max(
                walsh_error,
                abs(sum(float(v) for v in a["native_orders"][u, :, q]) - literal),
            )
    head_error = float(
        np.max(abs(states.astype(float) @ head.astype(float).T - logits))
    )
    embedding_action = np.einsum(
        "uk,ukd->ud",
        z["weights"][:, 0] - z["weights"][:, 1],
        e[50:].reshape(12, 16, -1),
    )
    result = {
        "PASS": max(source_error, action_error, balance_error, walsh_error) < 1e-8
        and head_error < 1e-3,
        "source_accounts": 12 * 16 * 5,
        "native_accounts": 12 * 16 * 2,
        "source_error": source_error,
        "action_error": action_error,
        "balance_error": balance_error,
        "walsh_error": walsh_error,
        "head_error": head_error,
        "additive_embedding_action_max": float(np.max(abs(embedding_action))),
        "source_answers_or_author_labels_used": False,
        "same_agent_alternate_not_blind": True,
        "all_original_scope_complete": False,
    }
    R.write(out / "verification.json", result)
    R.seal(out, "verification_seal.json", ("verification.json",))
    if not result["PASS"]:
        raise ValueError("Finite-account verification failed; no tolerance rescue")
    print(json.dumps(result), flush=True)


def main():
    """Expose immutable finite stages rather than an unbounded optimizer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "source", "design", "native", "analyze", "verify")
    )
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage in ("source", "native"):
        observe(args.out, args.stage)
    else:
        {"prepare": prepare, "design": design, "analyze": analyze, "verify": verify}[
            args.stage
        ](args.out)


if __name__ == "__main__":
    main()
