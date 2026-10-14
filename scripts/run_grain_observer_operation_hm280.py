#!/usr/bin/env python3
"""One same-evidence author/sample operation experiment, never a prompt grid."""

from __future__ import annotations

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
from grain_research.construction import math_behavior_predictive as B
from grain_research.construction import math_criterion_reader as C
from grain_research.construction import math_item_bridge as I
from grain_research.construction import math_observer_operation as M
from grain_research.construction import math_same_regime_reader as O
from scripts import run_grain_joint_persona_style_hm150 as J
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_prospective_evidence_hm218 as H

CONFIG = "configs/grain_observer_operation_hm280.json"
DEFAULT = ROOT / "results/grain_math_response_v1/observer_operation_r1"
TARGETS = ("E", "N", "A", "C", "O")
SCIENCE = (
    CONFIG,
    "docs/GRAIN_OBSERVER_OPERATION_PLAN_HM280.md",
    "grain_research/construction/math_observer_operation.py",
    "grain_research/construction/math_item_bridge.py",
    "grain_research/construction/math_behavior_predictive.py",
    "grain_research/construction/math_criterion_reader.py",
    "grain_research/construction/math_same_regime_reader.py",
    "grain_research/vector_mechanisms/trace.py",
    "scripts/run_grain_observer_operation_hm280.py",
    "scripts/score_grain_observer_operation_cold.py",
    "tests/test_grain_observer_operation.py",
)


def load(path):
    """Read numerical derivatives only, with pickle disabled."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def prepare(out):
    """Bind old evidence tokens and two finite operations before collecting outputs."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("No overwrite of scientific or failed outputs")
    cfg = R.read(ROOT / CONFIG)
    reader, criterion = (ROOT / cfg[k + "_parent"] for k in ("reader", "criterion"))
    old = R.check(reader, ("collected_seal.json", "analyzed_seal.json"))
    R.check(criterion, ("computed_seal.json", "verification_seal.json"))
    tok = AutoTokenizer.from_pretrained(str(J.MODEL), local_files_only=True)
    bank, cases = R.read(reader / "bank.json"), R.read(reader / "cases.json")
    old_start = B.observation_parts(
        tok, [tok.encode("x", add_special_tokens=False)[0]]
    )[0][:-1]
    ids = tok.apply_chat_template(
        [
            {"role": "system", "content": M.SYSTEM},
            {"role": "user", "content": "Writing sample:\n"},
        ],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
    )
    end = tok.convert_tokens_to_ids("<|im_end|>")
    pos = max(i for i, t in enumerate(ids) if t == end)
    start, tail = (
        ids[:pos],
        ids[pos:] + tok.encode("<|im_start|>assistant\n", add_special_tokens=False),
    )
    alphabet = [tok.encode(str(i), add_special_tokens=False) for i in range(2)]
    if (
        any(len(a) != 1 for a in alphabet)
        or alphabet[0] == alphabet[1]
        or len(bank) != 50
        or len(cases) != 48
    ):
        raise ValueError(
            "Fixed binary alphabet and complete old evidence/items required"
        )
    rows, lengths = [], []
    for case in cases + [
        {
            "unit_id": -1,
            "prefixes": [
                old_start
                + tok.encode(
                    "No writing sample is available.", add_special_tokens=False
                )
            ],
        }
    ]:
        for operation in M.OPERATIONS:
            for context, prefix in enumerate(case["prefixes"]):
                if prefix[: len(old_start)] != old_start:
                    raise ValueError("Archived neutral evidence prefix changed")
                body = prefix[len(old_start) :]
                sequences = [
                    start
                    + body
                    + tok.encode(
                        "\n\nEvidence ends here.\n\nQuestion: "
                        + M.question(b["public_text"], operation),
                        add_special_tokens=False,
                    )
                    + tail
                    for b in bank
                ]
                lengths.extend(map(len, sequences))
                rows.append(
                    {
                        "unit": case["unit_id"],
                        "operation": operation,
                        "context": context,
                        "input_ids": sequences,
                    }
                )
    if (
        len(rows) != cfg["families"]
        or len(rows) * 50 != cfg["inquiries"]
        or max(lengths) + 1 > cfg["max_input_tokens"]
    ):
        raise ValueError("Full registered support exceeds cap; no clipping/subsetting")
    model_file = J.MODEL / "model.safetensors"
    if not model_file.exists() or R.digest(model_file) != old["model_weights_sha256"]:
        raise ValueError("Pinned local model weights changed")
    inputs = [
        reader / n
        for n in (
            "manifest.json",
            "cases.json",
            "bank.json",
            "event_bindings.json",
            "numeric_pairs.npz",
            "observed_sums.npz",
            "collected_seal.json",
            "analyzed_seal.json",
        )
    ]
    inputs += [
        criterion / n
        for n in (
            "manifest.json",
            "readings.npz",
            "inference.npz",
            "computed_seal.json",
            "verification_seal.json",
        )
    ]
    inputs += [
        J.MODEL / n
        for n in (
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        )
    ]
    (out / "batches").mkdir(parents=True)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "alphabet": [a[0] for a in alphabet],
            "model_weights_sha256": old["model_weights_sha256"],
            "original_scope_complete": False,
        },
    )
    R.write(out / "units.json", rows)
    R.write(out / "bank.json", bank)
    R.write(out / "event_bindings.json", R.read(reader / "event_bindings.json"))
    quality = {
        "pairs": 48,
        "authors": 24,
        "families": len(rows),
        "inquiries": len(rows) * 50,
        "input_token_range": [min(lengths), max(lengths)],
        "raw_private_text_and_questionnaires_read": 0,
        "archived_public_evidence_tokens_unchanged": True,
        "empty_controls_not_missing_human_scores": True,
    }
    R.write(out / "input_quality.json", quality)
    R.seal(
        out,
        "prepared_seal.json",
        ("units.json", "bank.json", "event_bindings.json", "input_quality.json"),
    )
    H.status(out, {"state": "PREPARED", "families": 0, "model_seconds": 0})
    print(json.dumps(quality), flush=True)


def collect(out):
    """Observe every registered family once with a nonrenewable cap and checkpoints."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    manifest = R.check(out)
    if R.read(out / "runtime.json")["state"] != "PREPARED":
        raise ValueError("A scientific collection cannot be retried")
    cfg, alphabet = manifest["config"], manifest["alphabet"]
    done, elapsed, guard = 0, 0.0, 0.0
    lock = out / "ACTIVE.lock"
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    try:
        torch.set_num_threads(2)
        observer = DecoderObserver.load(
            str(J.MODEL),
            device=cfg["device"],
            dtype=cfg["dtype"],
            max_tokens=cfg["max_input_tokens"],
        )
        head = observer.model.lm_head
        R.archive(
            out / "head.npz",
            weights=head.weight.detach()[alphabet].float().cpu().numpy(),
            bias=np.zeros(2)
            if head.bias is None
            else head.bias.detach()[alphabet].float().cpu().numpy(),
        )
        rows = R.read(out / "units.json")
        for i, row in enumerate(rows):
            if elapsed >= cfg["max_model_seconds"]:
                raise TimeoutError("Single1500-model-second cap exhausted")
            H.status(
                out,
                {
                    "state": "RUNNING",
                    "pid": os.getpid(),
                    "pending_family": i,
                    "families": done,
                    "model_seconds": elapsed,
                },
            )
            t = time.monotonic()
            try:
                result = I.native_items(
                    observer.model,
                    row["input_ids"],
                    alphabet,
                    cfg["device"],
                    cfg["batch_size"],
                )
                if i == 0:
                    with torch.inference_mode():
                        for j in (0, 49):
                            ids = torch.tensor(
                                [row["input_ids"][j]], device=cfg["device"]
                            )
                            h = observer.model.model(
                                input_ids=ids, use_cache=False, return_dict=True
                            ).last_hidden_state[0, -1]
                            z = head(h).float().cpu().numpy()[alphabet]
                            guard = max(
                                guard,
                                float(
                                    abs(
                                        h.float().cpu().numpy() - result["states"][j]
                                    ).max()
                                ),
                                float(abs(z - result["logits"][j]).max()),
                            )
                    R.write(
                        out / "cache_guard.json",
                        {
                            "items": [0, 49],
                            "maximum_error": guard,
                            "tolerance": cfg["native_tolerance"],
                        },
                    )
                    if guard > cfg["native_tolerance"] or not np.isfinite(guard):
                        R.archive(out / "guard_failed.npz", **result)
                        raise ValueError("Cache guard failed; no precision rescue")
                R.archive(out / "batches" / f"{i:03d}.npz", **result)
                torch.mps.synchronize()
            finally:
                elapsed += time.monotonic() - t
            done += 1
            if done % 32 == 0:
                print(
                    f"HM280 {done}/{len(rows)} families {elapsed:.3f}model-seconds",
                    flush=True,
                )
        runtime = {
            "state": "COLLECTED",
            "pid": None,
            "families": done,
            "inquiries": done * 50,
            "model_seconds": elapsed,
            "guard_error": guard,
            "new_API_private_training": 0,
        }
        R.write(out / "collected_runtime.json", runtime)
        R.seal(
            out,
            "collected_seal.json",
            tuple(f"batches/{i:03d}.npz" for i in range(done))
            + ("head.npz", "cache_guard.json", "collected_runtime.json"),
        )
        H.status(out, runtime)
        print(json.dumps(runtime), flush=True)
    except BaseException as exc:
        H.status(
            out,
            {
                "state": "FAILED",
                "pid": None,
                "families": done,
                "model_seconds": elapsed,
                "error": repr(exc),
                "no_auto_retry": True,
            },
        )
        raise
    finally:
        lock.unlink(missing_ok=True)


def responses(out):
    """Bind complete observations to pair, operation and context rather than row order alone."""
    values = np.empty((48, 2, 4, 50, 2))
    empty = np.empty((2, 50, 2))
    diagnostics, head_error = [], 0.0
    head = load(out / "head.npz")
    alphabet = R.read(out / "manifest.json")["alphabet"]
    for i, row in enumerate(R.read(out / "units.json")):
        r = load(out / "batches" / f"{i:03d}.npz")
        head_error = max(
            head_error,
            float(
                abs(
                    M.head_logits(r["states"], head["weights"], head["bias"])
                    - r["logits"]
                ).max()
            ),
        )
        o = M.OPERATIONS.index(row["operation"])
        if row["unit"] < 0:
            empty[o] = r["logits"]
        else:
            values[row["unit"], o, row["context"]] = r["logits"]
        logmass = logsumexp(r["logits"], axis=-1) - r["normalizer"]
        if np.max(logmass) > 1e-6:
            raise ArithmeticError("Binary candidate FULL-vocabulary mass exceeds one")
        diagnostics.extend(
            {
                "family": i,
                "item": j,
                "operation": row["operation"],
                "binary_log_mass": float(logmass[j]),
                "greedy_id": int(r["top"][j]),
                "greedy_in_alphabet": bool(r["top"][j] in alphabet),
            }
            for j in range(50)
        )
    return values, empty, diagnostics, head_error


def analyze(out):
    """Keep both observation laws and all signed SOURCE contact accounts; fit nothing."""
    cfg = R.check(out, ("collected_seal.json",))["config"]
    z, empty, diagnostics, error = responses(out)
    if error > cfg["native_tolerance"]:
        raise ArithmeticError("Cold head fails adequate fixed tolerance")
    bank = R.read(out / "bank.json")
    accounts, blanks = M.operation_accounts(z, bank), M.key_values(empty, bank)
    old = load(ROOT / cfg["criterion_parent"] / "readings.npz")
    reference = load(ROOT / cfg["criterion_parent"] / "inference.npz")
    arrays, rows, primary, operation_summary = {}, [], [], {}
    for scale, keys in accounts.items():
        for key, account in keys.items():
            values = account["values"]
            arrays[f"{scale}/{key}"] = values
            for o, operation in enumerate(M.OPERATIONS + ("operation_difference",)):
                reader = values[:, o] if o < 2 else values[:, 0] - values[:, 1]
                arrays[f"{scale}/{key}/{operation}/parts"] = np.stack(
                    list(O.reader_parts(reader).values())
                )
                for sp, space in enumerate(("E5", "GRAIN15")):
                    for b, source_bank in enumerate(("bank0", "bank1", "pooled")):
                        for win, mask in enumerate(
                            (
                                old["windows"] == 0,
                                old["windows"] == 1,
                                np.ones(48, bool),
                            )
                        ):
                            contact = C.contacts(
                                old["scores"][sp, b, mask], reader[mask]
                            )
                            for component, component_name in enumerate(
                                ("total", "shared", "within")
                            ):
                                for part, part_name in enumerate(
                                    ("join", "order", "single", "raw")
                                ):
                                    for q in range(5):
                                        for r in range(5):
                                            rows.append(
                                                {
                                                    "scale": scale,
                                                    "key": key,
                                                    "operation": operation,
                                                    "space": space,
                                                    "source_bank": source_bank,
                                                    "window": win,
                                                    "component": component_name,
                                                    "part": part_name,
                                                    "source_target": TARGETS[q],
                                                    "reader_target": TARGETS[r],
                                                    "contact": float(
                                                        contact[component, part, q, r]
                                                    ),
                                                }
                                            )
                            if key == "balanced_key" and b == 2 and win == 2:
                                for q in range(5):
                                    primary.append(
                                        {
                                            "scale": scale,
                                            "operation": operation,
                                            "space": space,
                                            "target": TARGETS[q],
                                            "total_join_contact": float(
                                                contact[0, 0, q, q]
                                            ),
                                            "shared_join_contact": float(
                                                contact[1, 0, q, q]
                                            ),
                                            "within_join_contact": float(
                                                contact[2, 0, q, q]
                                            ),
                                        }
                                    )
            if key == "balanced_key":
                parts = account["parts"]
                operation_summary[scale] = {
                    "joint_RMS_by_operation": {
                        o: np.sqrt(np.mean(parts[o]["join"] ** 2, axis=0)).tolist()
                        for o in parts
                    },
                    "level_difference_RMS": np.sqrt(
                        np.mean((values[:, 0] - values[:, 1]) ** 2, axis=(0, 1))
                    ).tolist(),
                    "empty_control_values": blanks[scale][key].tolist(),
                }
                alpha = np.stack(
                    [
                        C.observer_pullback(
                            old["weights"][sp, 2], values[:, 0] - values[:, 1]
                        )
                        for sp in range(2)
                    ]
                )
                observed, null, variance, p = C.conditional_reference(
                    alpha,
                    old["labels"],
                    old["means"],
                    old["groups"],
                    reference["assignments"],
                )
                arrays[f"{scale}/operation_reference/observed"] = observed
                arrays[f"{scale}/operation_reference/p"] = p
                arrays[f"{scale}/operation_reference/null"] = null
                arrays[f"{scale}/operation_reference/variance"] = variance
                operation_summary[scale]["within_contact_difference_matching_p"] = (
                    np.diagonal(p, axis1=1, axis2=2).tolist()
                )
    R.archive(out / "accounts.npz", **arrays)
    R.archive(out / "logits.npz", values=z, empty=empty)
    R.table(out / "all_contacts.csv", rows)
    R.table(out / "primary.csv", primary)
    R.table(out / "vocabulary_diagnostics.csv", diagnostics)
    R.write(
        out / "summary.json",
        {
            "operations": operation_summary,
            "all_contacts": len(rows),
            "head_error": error,
            "greedy_in_alphabet_rate": float(
                np.mean([r["greedy_in_alphabet"] for r in diagnostics])
            ),
            "native_collection": R.read(out / "collected_runtime.json"),
            "no_accuracy_fit_sign_rescue": True,
            "not_actual_human_trait_or_state_answers": True,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "accounts.npz",
            "logits.npz",
            "all_contacts.csv",
            "primary.csv",
            "vocabulary_diagnostics.csv",
            "summary.json",
        ),
    )
    print(json.dumps(R.read(out / "summary.json")), flush=True)


def verify(out):
    """Use literal item loops and contacts to check the existing observations, not rerun them."""
    cfg = R.check(out, ("collected_seal.json", "analyzed_seal.json"))["config"]
    z, _, _, head_error = responses(out)
    bank = R.read(out / "bank.json")
    accounts, key_error, contact_error = load(out / "accounts.npz"), 0.0, 0.0
    from math import exp

    for scale in ("probability", "log_odds"):
        literal = np.empty((48, 2, 4, 5))
        for u in range(48):
            for o in range(2):
                for c in range(4):
                    for q in range(5):
                        poles = [
                            [
                                i
                                for i, b in enumerate(bank)
                                if b["target"] == q and b["pole"] == p
                            ]
                            for p in (1, -1)
                        ]
                        means = []
                        for indices in poles:
                            values = []
                            for j in indices:
                                odds = float(z[u, o, c, j, 1] - z[u, o, c, j, 0])
                                values.append(
                                    1 / (1 + exp(-odds))
                                    if scale == "probability"
                                    else odds
                                )
                            means.append(sum(values) / len(values))
                        literal[u, o, c, q] = (means[0] - means[1]) / 2 + (
                            0.5 if scale == "probability" else 0
                        )
        key_error = max(
            key_error, float(abs(literal - accounts[f"{scale}/balanced_key"]).max())
        )
        old = load(ROOT / cfg["criterion_parent"] / "readings.npz")
        for sp in range(2):
            src = old["scores"][sp, 2]
            for o in range(3):
                reading = literal[:, o] if o < 2 else literal[:, 0] - literal[:, 1]
                joint = (
                    reading[:, 0] + reading[:, 1] - reading[:, 2] - reading[:, 3]
                ) / 2
                for q in range(5):
                    for r in range(5):
                        actual = (
                            sum(
                                ((src[i, 0, 0, q] + src[i, 1, 0, q]) / 2) * joint[i, r]
                                for i in range(48)
                            )
                            / 48
                        )
                        expected = O.contacts(src[:, :, 0], reading)["join"][q, r]
                        contact_error = max(
                            contact_error, abs(float(actual - expected))
                        )
    report = {
        "PASS": head_error < cfg["native_tolerance"]
        and max(key_error, contact_error) < 1e-12,
        "head_error": head_error,
        "literal_key_error": key_error,
        "literal_contact_error": contact_error,
        "same_agent_not_blind_audit": True,
        "no_new_neural_observations": True,
    }
    if not report["PASS"]:
        raise ArithmeticError(json.dumps(report))
    R.write(out / "verification.json", report)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(report), flush=True)


def main():
    """Run fixed finite stages without restarting an existing scientific collection."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage", choices=("prepare", "collect", "analyze", "verify", "check")
    )
    p.add_argument("--output", type=Path, default=DEFAULT)
    a = p.parse_args()
    if a.stage == "check":
        R.check(
            a.output,
            ("collected_seal.json", "analyzed_seal.json", "verification_seal.json"),
        )
        print("HM280 seals PASS")
    else:
        globals()[a.stage](a.output)


if __name__ == "__main__":
    main()
