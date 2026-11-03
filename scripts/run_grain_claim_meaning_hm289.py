#!/usr/bin/env python3
"""One finite SOURCE-qualified claim involution;no accuracy or precision search."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_claim_meaning as M
from grain_research.construction import math_criterion_components as C
from grain_research.construction import math_person_bound_response as O
from grain_research.construction import math_person_bound_tree as T
from grain_research.construction import math_source_anchor_exchange as E
from scripts import run_grain_joint_persona_style_hm150 as J
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_prospective_evidence_hm218 as H
from scripts.score_grain_author_belief_cold import load_bundle

CONFIG = "configs/grain_claim_meaning_hm289.json"
SPEC = R.read(ROOT / CONFIG)
DEFAULT = ROOT / SPEC["output"]
TARGETS = ("E", "N", "A", "C", "O")
SCIENCE = (
    CONFIG,
    "docs/GRAIN_CLAIM_MEANING_PLAN_HM289.md",
    "grain_research/construction/math_claim_meaning.py",
    "scripts/run_grain_claim_meaning_hm289.py",
    "scripts/score_grain_claim_meaning_cold.py",
    "tests/test_grain_claim_meaning.py",
    "grain_research/construction/math_person_bound_tree.py",
    "grain_research/construction/math_person_bound_response.py",
    "grain_research/construction/math_behavior_predictive.py",
    "grain_research/construction/math_criterion_components.py",
    "grain_research/construction/math_source_anchor_exchange.py",
    "grain_research/construction/math_item_bridge.py",
    "grain_research/vector_mechanisms/trace.py",
    "scripts/run_grain_joint_persona_style_hm150.py",
    "scripts/run_grain_observed_family_hm170.py",
    "scripts/run_grain_prospective_evidence_hm218.py",
    "scripts/score_grain_author_belief_cold.py",
)


def prepare(out):
    """Compile both new claim frames without touching old bodies or SOURCE laws."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("No overwrite of a scientific packet")
    material, tree = ROOT / SPEC["material_parent"], ROOT / SPEC["tree_parent"]
    stages = ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")
    parent = R.check(material, stages)
    R.check(tree, stages)
    tok = AutoTokenizer.from_pretrained(str(J.MODEL), local_files_only=True)
    encode = lambda text: tok.encode(text, add_special_tokens=False)
    tail = encode("<|im_end|>\n<|im_start|>assistant\n")
    boundary = "\nEvidence ends here.\n\nQuestion: "
    alphabet = [encode(str(j)) for j in range(3)]
    if any(len(a) != 1 for a in alphabet) or len({a[0] for a in alphabet}) != 3:
        raise ValueError("All three registered single-token digits required")
    if [a[0] for a in alphabet[:2]] != parent["alphabet"]:
        raise ValueError("Old binary digits cannot be silently replaced")
    bank, old_rows = R.read(material / "bank.json"), R.read(material / "units.json")
    rows, lengths = [], []
    for old_row in old_rows:
        for polarity in (1, -1):
            sequences = []
            for item, old in zip(bank, old_row["input_ids"], strict=True):
                suffix = (
                    encode(
                        boundary + O.question(item["public_text"], old_row["queried"])
                    )
                    + tail
                )
                if old[-len(suffix) :] != suffix:
                    raise ValueError("Original complete evidence boundary changed")
                prefix = old[: -len(suffix)]
                new = (
                    encode(
                        boundary
                        + M.question(item["public_text"], old_row["queried"], polarity)
                    )
                    + tail
                )
                sequences.append(prefix + new)
                if sequences[-1][: -len(new)] != prefix:
                    raise ValueError("Actual comment token prefix changed")
            lengths.extend(map(len, sequences))
            rows.append(
                {k: v for k, v in old_row.items() if k != "input_ids"}
                | {
                    "polarity": polarity,
                    "input_ids": sequences,
                }
            )
    for unit in range(48):
        for queried in (-1, 1):
            for polarity in (-1, 1):
                bags = [
                    r["input_ids"][0]
                    for r in rows
                    if r["unit"] == unit
                    and r["queried"] == queried
                    and r["polarity"] == polarity
                ]
                if len(bags) != 4:
                    raise ValueError("Full ownership/display cube required")
                O.assert_same_bag(bags)
    if (
        len(rows) != SPEC["families"]
        or len(rows) * 50 != SPEC["inquiries"]
        or max(lengths) > SPEC["max_input_tokens"]
    ):
        raise ValueError("Full support exceeds registered cap;no clipping/subset")
    if R.digest(J.MODEL / "model.safetensors") != parent["model_weights_sha256"]:
        raise ValueError("Pinned observer changed")
    inputs = [
        material / p
        for p in (
            "units.json",
            "bank.json",
            "cases.json",
            "manifest.json",
            "verification_seal.json",
        )
    ]
    inputs += [
        tree / p
        for p in (
            "source.npz",
            "readings.npz",
            "summary.json",
            "manifest.json",
            "verification_seal.json",
        )
    ]
    inputs += [
        J.MODEL / p
        for p in (
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
            "config": SPEC,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "alphabet": [a[0] for a in alphabet],
            "model_weights_sha256": parent["model_weights_sha256"],
            "matched_new_positive_not_old_HM288_positive": True,
            "original_scope_complete": False,
        },
    )
    R.write(out / "units.json", rows)
    R.write(out / "bank.json", bank)
    R.write(out / "cases.json", R.read(material / "cases.json"))
    R.archive(out / "source.npz", **load_bundle(tree / "source.npz"))
    R.write(
        out / "input_quality.json",
        {
            "authors": 24,
            "comments": 96,
            "dependent_dyads": 48,
            "families": len(rows),
            "inquiries": len(rows) * 50,
            "input_token_range": [min(lengths), max(lengths)],
            "complete_original_evidence_prefixes_preserved": True,
            "new_API_private_training_answer_fit_grid": 0,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        ("units.json", "bank.json", "cases.json", "source.npz", "input_quality.json"),
    )
    H.status(out, {"state": "PREPARED", "families": 0, "model_seconds": 0})
    print(json.dumps(R.read(out / "input_quality.json")), flush=True)


def collect(out):
    """Capture each registered endpoint once;use alphabet-sized absent head bias."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    manifest = R.check(out)
    if R.read(out / "runtime.json")["state"] != "PREPARED":
        raise ValueError("No scientific collection retries or renewed budgets")
    alphabet, cfg = manifest["alphabet"], manifest["config"]
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
        bias = (
            None
            if head.bias is None
            else head.bias.detach()[alphabet].float().cpu().numpy()
        )
        R.archive(
            out / "head.npz",
            weights=head.weight.detach()[alphabet].float().cpu().numpy(),
            bias=M.exported_bias(len(alphabet), bias),
        )
        rows = R.read(out / "units.json")
        for i, row in enumerate(rows):
            if elapsed >= cfg["max_model_seconds"]:
                raise TimeoutError("One nonrenewable4800-model-second cap exhausted")
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
            start = time.monotonic()
            try:
                result = T.native_items(
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
                    if not np.isfinite(guard) or guard > cfg["native_tolerance"]:
                        R.archive(out / "guard_failed.npz", **result)
                        raise ValueError(
                            "Native cache guard failed;no precision rescue"
                        )
                R.archive(out / "batches" / f"{i:03d}.npz", **result)
                if cfg["device"] == "mps":
                    torch.mps.synchronize()
            finally:
                elapsed += time.monotonic() - start
            done += 1
            if elapsed > cfg["max_model_seconds"]:
                raise TimeoutError("Registered budget crossed;partial data retained")
            if done % 32 == 0:
                print(
                    f"HM289 {done}/{len(rows)} families {elapsed:.3f}model-seconds",
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


def observed(out):
    """Read all actual signed claim endpoints and vocabulary diagnostics."""
    z = np.empty((2, 48, 2, 2, 2, 50, 3))
    seen, diagnostics, error = set(), [], 0.0
    head, alphabet = (
        load_bundle(out / "head.npz"),
        R.read(out / "manifest.json")["alphabet"],
    )
    M.exported_bias(3, head["bias"])
    for i, row in enumerate(R.read(out / "units.json")):
        key = (
            0 if row["polarity"] == 1 else 1,
            row["unit"],
            (row["ownership"] + 1) // 2,
            (row["queried"] + 1) // 2,
            (row["display"] + 1) // 2,
        )
        if key in seen:
            raise ValueError("Duplicate claim/event observation")
        seen.add(key)
        packet = load_bundle(out / "batches" / f"{i:03d}.npz")
        z[key] = packet["logits"]
        error = max(
            error,
            float(
                abs(
                    packet["states"] @ head["weights"].astype(float).T
                    + head["bias"]
                    - packet["logits"]
                ).max()
            ),
        )
        mass = logsumexp(packet["logits"], -1) - packet["normalizer"]
        if not np.isfinite(mass).all() or max(mass) > 1e-6:
            raise ArithmeticError("Registered alphabet is not a vocabulary subset")
        diagnostics.extend(
            {
                "family": i,
                "polarity": row["polarity"],
                "item": j,
                "log_category_mass": float(mass[j]),
                "greedy_category": alphabet.index(int(packet["top"][j]))
                if int(packet["top"][j]) in alphabet
                else -1,
            }
            for j in range(50)
        )
    if seen != set(np.ndindex(2, 48, 2, 2, 2)) or error > SPEC["native_tolerance"]:
        raise ArithmeticError("Missing observations or native head replay mismatch")
    return z, error, diagnostics


def analyze(out):
    """Account for both sectors with unchanged SOURCE law,not a fitted reader."""
    R.check(out, ("collected_seal.json",))
    z, error, diagnostics = observed(out)
    source, bank = load_bundle(out / "source.npz"), R.read(out / "bank.json")
    fields, sides, trees = M.readings(z, bank)
    tree_parent = ROOT / SPEC["tree_parent"]
    old_fields, _ = T.readings(
        load_bundle(tree_parent / "readings.npz")["logits"], bank
    )
    for channel in T.CHANNELS:
        fields["positive_frame_change_" + channel] = {
            key: sides[0][channel][key] - old_fields[channel][key]
            for key in old_fields[channel]
        }
    arrays, rows = {"logits": z}, []
    for channel, values in fields.items():
        arrays[channel + "/items"] = values["items"]
        for key in ("balanced_key", "common_support", "global_common"):
            contact = np.stack(
                [
                    np.stack([T.contacts(s, values[key]) for s in space])
                    for space in source["components"]
                ]
            )
            arrays[channel + "/" + key] = contact
            if key == "global_common":
                continue
            for sp, ref, law, component, mask, q, r in np.ndindex(contact.shape):
                rows.append(
                    {
                        "space": ("E5", "GRAIN15")[sp],
                        "reference": ("bank0", "bank1", "pooled")[ref],
                        "law": ("F", "J", "F_minus_J")[law],
                        "component": ("total", "composition", "within_configuration")[
                            component
                        ],
                        "channel": channel,
                        "character": mask,
                        "source_target": TARGETS[q],
                        "query": TARGETS[r],
                        "readout": key,
                        "contact": float(contact[sp, ref, law, component, mask, q, r]),
                    }
                )
    e = source["labels"] - source["means"]
    primary, alphas, nulls = [], [], []
    for channel in M.PRIMARY:
        gamma = fields[channel]["global_common"][:, 3]
        alpha = source["relative_weights"][1, 2, 0].T @ gamma / 48
        alphas.append(alpha)
        channel_nulls = []
        for q, target in enumerate(TARGETS):
            value = float(alpha @ e[:, q])
            null = e[source["assignments"][q], q] @ alpha
            primary.append(
                {
                    "channel": channel,
                    "target": target,
                    "source_person_bound_common": value,
                    "exact_conditional_mean": 0.0,
                    "exact_conditional_variance": float(
                        alpha
                        @ C.orbit_covariance(
                            source["labels"][:, q], source["groups"][:, q]
                        )
                        @ alpha
                    ),
                    "two_sided_conditional_rank": float(
                        (1 + np.sum(abs(null) >= abs(value))) / 200
                    ),
                }
            )
            channel_nulls.append(null)
        nulls.append(channel_nulls)
    for row, adjusted in zip(
        primary, E.holm([p["two_sided_conditional_rank"] for p in primary]), strict=True
    ):
        row["Holm10_rank"] = float(adjusted)
    record = {
        "version": "HM289-v1",
        "cases": R.read(out / "cases.json"),
        "source_components": source["components"][1, 2].tolist(),
        "item_logits": z.tolist(),
        "item_bank": [{k: b[k] for k in ("item", "target", "pole")} for b in bank],
    }
    R.archive(
        out / "readings.npz",
        **arrays,
        alphas=np.asarray(alphas),
        nulls=np.asarray(nulls),
        log_probabilities=np.stack([t["log_probabilities"] for t in trees]),
    )
    R.table(out / "all_contacts.csv", rows)
    R.table(out / "primary.csv", primary)
    R.table(out / "reader_diagnostics.csv", diagnostics)
    R.write(out / "cold_record.json", record)
    R.write(
        out / "summary.json",
        {
            "primary": primary,
            "head_error": error,
            "all8_global_common_RMS": {
                key: np.sqrt(np.mean(value["global_common"] ** 2, axis=0)).tolist()
                for key, value in fields.items()
            },
            "positive_negative_category_probability_means": [
                np.exp(t["log_probabilities"]).mean(axis=(0, 1, 2, 3, 4)).tolist()
                for t in trees
            ],
            "positive_negative_greedy_counts": [
                [
                    sum(
                        d["polarity"] == p and d["greedy_category"] == j
                        for d in diagnostics
                    )
                    for j in (0, 1, 2, -1)
                ]
                for p in (1, -1)
            ],
            "native_runtime": R.read(out / "collected_runtime.json"),
            "not_human_truth_or_accuracy_gain": True,
            "symmetry_prediction_not_imposed": True,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "readings.npz",
            "all_contacts.csv",
            "primary.csv",
            "reader_diagnostics.csv",
            "cold_record.json",
            "summary.json",
        ),
    )
    print(json.dumps(R.read(out / "summary.json")), flush=True)


def verify(out):
    """Check independent literal involution/role sums,source contacts and cold replay."""
    R.check(out, ("collected_seal.json", "analyzed_seal.json"))
    data, source = load_bundle(out / "readings.npz"), load_bundle(out / "source.npz")
    summary, bank = R.read(out / "summary.json"), R.read(out / "bank.json")
    z = data["logits"]
    d, a = z[..., 1] - z[..., 0], np.logaddexp(z[..., 0], z[..., 1]) - z[..., 2]
    raw = {
        "semantic_direction": (d[0] - d[1]) / 2,
        "affirmation_direction": (d[0] + d[1]) / 2,
        "availability_even": (a[0] + a[1]) / 2,
        "availability_odd": (a[0] - a[1]) / 2,
    }
    errors = {"source_references": 0.0, "variance": 0.0}
    for name, value in raw.items():
        literal = np.zeros((48, 8, 50))
        for mask in range(8):
            for corner in np.ndindex(2, 2, 2):
                signs = [2 * j - 1 for j in corner]
                weight = np.prod([signs[k] for k in range(3) if mask & (1 << k)])
                literal[:, mask] += weight * value[(slice(None), *corner)] / 8
        errors[name + "/characters"] = float(abs(literal - data[name + "/items"]).max())
        for key in ("balanced_key", "common_support", "global_common"):
            c = data[name + "/" + key]
            errors[name + "/" + key + "/levels"] = float(
                abs(c[:, :, 0] - c[:, :, 1] - c[:, :, 2]).max()
            )
            errors[name + "/" + key + "/parts"] = float(
                abs(c[:, :, :, 0] - c[:, :, :, 1] - c[:, :, :, 2]).max()
            )
        if name not in M.PRIMARY:
            continue
        hi = [
            [b["item"] for b in bank if b["target"] == q and b["pole"] == 1]
            for q in range(5)
        ]
        lo = [
            [b["item"] for b in bank if b["target"] == q and b["pole"] == -1]
            for q in range(5)
        ]
        gamma = (
            sum(
                (
                    sum(literal[:, 3, j] for j in hi[q]) / len(hi[q])
                    + sum(literal[:, 3, j] for j in lo[q]) / len(lo[q])
                )
                / 2
                for q in range(5)
            )
            / 5
        )
        idx = M.PRIMARY.index(name)
        with threadpool_limits(limits=1):
            for q in range(5):
                e = source["labels"][:, q] - source["means"][:, q]
                saved = np.r_[
                    summary["primary"][idx * 5 + q]["source_person_bound_common"],
                    data["nulls"][idx, q],
                ]
                for assignment, expected in zip(
                    np.vstack((np.arange(386), source["assignments"][q])),
                    saved,
                    strict=True,
                ):
                    if not np.array_equal(
                        source["groups"][assignment, q], source["groups"][:, q]
                    ):
                        raise ArithmeticError("Old SOURCE assignment law changed")
                    scores = np.array(
                        [
                            sum(w * e[assignment])
                            for w in source["relative_weights"][1, 2, 0]
                        ]
                    )
                    errors["source_references"] = max(
                        errors["source_references"],
                        abs(sum(scores * gamma) / 48 - expected),
                    )
                variance = 0.0
                for group in np.unique(source["groups"][:, q]):
                    ix = source["groups"][:, q] == group
                    y, ell = source["labels"][ix, q], data["alphas"][idx, ix]
                    n, p = len(y), y.mean()
                    if n > 1:
                        variance += (
                            p
                            * (1 - p)
                            * n
                            / (n - 1)
                            * (sum(ell**2) - sum(ell) ** 2 / n)
                        )
                errors["variance"] = max(
                    errors["variance"],
                    abs(
                        variance
                        - summary["primary"][idx * 5 + q]["exact_conditional_variance"]
                    ),
                )
    fields, sides, trees = M.readings(z, bank)
    old_fields, _ = T.readings(
        load_bundle(ROOT / SPEC["tree_parent"] / "readings.npz")["logits"], bank
    )
    for key in ("items", "balanced_key", "common_support", "global_common"):
        for sign, side in ((1, 0), (-1, 1)):
            errors[f"direction_reconstruction/{key}/{side}"] = float(
                abs(
                    fields["affirmation_direction"][key]
                    + sign * fields["semantic_direction"][key]
                    - sides[side]["direction"][key]
                ).max()
            )
        for channel in T.CHANNELS:
            expected = sides[0][channel][key] - old_fields[channel][key]
            saved = data["positive_frame_change_" + channel + "/" + key]
            if key != "items":
                expected = np.stack(
                    [
                        np.stack([T.contacts(s, expected) for s in space])
                        for space in source["components"]
                    ]
                )
            errors[f"old_frame/{channel}/{key}"] = float(abs(expected - saved).max())
    for side, tree in enumerate(trees):
        errors[f"categories/{side}"] = float(
            abs(
                T.tree_log_probabilities(tree["direction"], tree["assessability"])
                - tree["log_probabilities"]
            ).max()
        )
    response = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_claim_meaning_cold.py"),
            "--input",
            str(out / "cold_record.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cold = json.loads(response.stdout)
    for channel in M.CHANNELS:
        for key in ("balanced_key", "common_support", "global_common"):
            errors[f"cold/{channel}/{key}"] = float(
                abs(
                    np.asarray(cold["channels"][channel][key])
                    - data[channel + "/" + key][1, 2]
                ).max()
            )
    if max(errors.values()) > 1e-10 or cold["neural_modules_loaded"]:
        raise ArithmeticError("Finite claim law/old account/cold replay failed")
    R.write(out / "cold_readback.json", cold)
    R.write(
        out / "verification.json",
        {
            "PASS": True,
            "errors": errors,
            "literal_primary_and_old_references": 2000,
            "same_agent_not_blind": True,
            "new_model_calls": 0,
        },
    )
    R.seal(out, "verification_seal.json", ("cold_readback.json", "verification.json"))
    print(json.dumps(R.read(out / "verification.json")), flush=True)


def main():
    """Run once-only finite stages;run continues only this new prepared packet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "collect", "analyze", "verify", "check", "run")
    )
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "run":
        collect(args.out)
        analyze(args.out)
        verify(args.out)
        R.check(
            args.out,
            ("collected_seal.json", "analyzed_seal.json", "verification_seal.json"),
        )
        print("HM289 finite scientific stages and seals PASS", flush=True)
    elif args.stage == "check":
        R.check(
            args.out,
            ("collected_seal.json", "analyzed_seal.json", "verification_seal.json"),
        )
        print("HM289 scientific seals PASS")
    else:
        {"prepare": prepare, "collect": collect, "analyze": analyze, "verify": verify}[
            args.stage
        ](args.out)


if __name__ == "__main__":
    main()
