#!/usr/bin/env python3
"""Finite prospective behavioral anchors for the sealed real-evidence blocks."""

import argparse
import hashlib
import os
import sys
import time
from collections.abc import Mapping
from itertools import product
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_beneficiary_choice as M
from grain_research.construction.math_prospective_evidence import bootstrap
from grain_research.vector_mechanisms.navigation import MODEL
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_prospective_evidence_hm218 as H
from scripts.run_grain_vector_mechanism_external_features import bundle_arms

CONFIG = "configs/grain_beneficiary_choice_hm220.json"
DEFAULT = ROOT / "results/grain_math_response_v1/beneficiary_choice_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_BENEFICIARY_CHOICE_PLAN_HM220.md",
    "grain_research/construction/math_beneficiary_choice.py",
    "scripts/run_grain_beneficiary_choice_hm220.py",
    "tests/test_grain_beneficiary_choice.py",
)


def prepare(out):
    """Freeze complete endpoint tokens and actual source-selected event binding."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve every existing or uncertain scientific run")
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["parent"]
    pm = R.check(
        parent, ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    previous = ROOT / pm["config"]["parent"]
    gm = R.check(
        previous,
        ("collected_seal.json", "analyzed_seal.json", "verification_seal.json"),
    )
    material = ROOT / gm["config"]["material"] / "private_material.json"
    bundles = R.read(material)["bundles"]
    panel = H.selected(parent / "observations.npz", ("panel_indices",))["panel_indices"]
    blocks = R.read(parent / "blocks.json")
    selections = R.read(previous / "selections.json")
    tok = AutoTokenizer.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False
    )
    codes = [tok.encode(k, add_special_tokens=False) for k in ("L", "R")]
    if any(len(row) != 1 for row in codes) or codes[0] == codes[1]:
        raise ValueError("Distinct single-token choice alphabet required")
    units, bindings, lengths = [], [], []
    for u, row in enumerate(blocks):
        if row["author"] != u:
            raise ValueError("Changed author addresses")
        bundle = bundles[int(panel[u])]
        event_ids = [e["comment_id"] for e in bundle["views"][2]]
        if event_ids != selections[u]["event_ids"]:
            raise ValueError("Actual event addresses changed")
        texts = bundle_arms(bundle)["cue_mask"][24:36]
        for arm, name in enumerate(("support", "counter")):
            slots = sorted(row[name])
            selected = [texts[j] for j in slots]
            if slots != sorted(selections[u][name]):
                raise ValueError("Source selection changed")
            bindings.append(
                {
                    "author": u,
                    "arm": arm,
                    "event_ids": [event_ids[j] for j in slots],
                    "body_hashes": [
                        hashlib.sha256(t.encode()).hexdigest() for t in selected
                    ],
                }
            )
            for b, c, o in product(range(2), repeat=3):
                user = M.render(selected, cfg["beneficiaries"][b], cfg["costs"][c], o)
                ids = tok.apply_chat_template(
                    [
                        {"role": "system", "content": M.SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                if isinstance(ids, Mapping):
                    ids = ids["input_ids"]
                if not isinstance(ids, list) or not all(
                    isinstance(i, int) and i >= 0 for i in ids
                ):
                    raise ValueError("Actual template token IDs required")
                if not ids or len(ids) + 16 > cfg["max_tokens"]:
                    raise ValueError(
                        "Unsupported input; do not truncate or omit author"
                    )
                units.append(
                    {
                        "author": u,
                        "arm": arm,
                        "beneficiary": b,
                        "cost": c,
                        "order": o,
                        "input_ids": ids,
                    }
                )
                lengths.append(len(ids))
    if len(blocks) != cfg["authors"] or len(units) != cfg["enquiries"]:
        raise ValueError("Incomplete finite scope")
    files = {
        p.name: R.digest(p)
        for p in MODEL.iterdir()
        if p.is_file() and not p.name.startswith("._")
    }
    if files != pm["model_files"]:
        raise ValueError("Changed inherited model")
    (out / "batches").mkdir(parents=True, exist_ok=False)
    R.write(out / "units.json", units)
    R.write(out / "bindings.json", bindings)
    R.write(
        out / "input_quality.json",
        {
            "authors": 24,
            "reused_event_addresses": 288,
            "new_endpoints": len(units),
            "prompt_tokens_range": [min(lengths), max(lengths)],
            "new_selections_application_labels_raw_processing": False,
            "no_input_truncation": True,
            "all_endpoints_before_answers": True,
        },
    )
    inputs = [
        parent / n
        for n in (
            "manifest.json",
            "blocks.json",
            "observations.npz",
            "verification_seal.json",
        )
    ]
    inputs += [previous / "selections.json", previous / "manifest.json", material]
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "alphabet": {"labels": ["L", "R"], "token_ids": [x[0] for x in codes]},
            "model_files": files,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "parent_results_known_new_choices_unobserved": True,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out, "prepared_seal.json", ("units.json", "bindings.json", "input_quality.json")
    )
    H.status(out, {"status": "PREPARED", "completed_batches": 0})
    print("PREPARED HM220:24authors/48batches/384newchoiceendpoints", flush=True)


def collect(out):
    """Collect each finite batch once with bounded cost and recoverable state."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    manifest = R.check(out)
    cfg = manifest["config"]
    if R.read(out / "runtime.json")["status"] != "PREPARED":
        raise ValueError("No automatic uncertain/terminal retry")
    for name, digest in manifest["model_files"].items():
        if R.digest(MODEL / name) != digest:
            raise ValueError("Changed observer")
    lock = out / "ACTIVE.lock"
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    elapsed, completed = 0.0, 0
    try:
        torch.set_num_threads(2)
        observer = DecoderObserver.load(
            str(MODEL),
            device=cfg["device"],
            dtype=cfg["dtype"],
            max_tokens=cfg["max_tokens"],
        )
        alpha = manifest["alphabet"]["token_ids"]
        R.archive(
            out / "head.npz",
            weights=observer.model.lm_head.weight.detach()[alpha].float().cpu().numpy(),
        )
        units = R.read(out / "units.json")
        for start in range(0, len(units), cfg["batch_size"]):
            batch, bi = (
                units[start : start + cfg["batch_size"]],
                start // cfg["batch_size"],
            )
            if len({(b["author"], b["arm"]) for b in batch}) != 1 or [
                (b["beneficiary"], b["cost"], b["order"]) for b in batch
            ] != list(product(range(2), repeat=3)):
                raise ValueError("Endpoint binding changed")
            if elapsed >= cfg["max_model_seconds"]:
                raise RuntimeError("Finite cap reached; preserve without extension")
            H.status(
                out,
                {
                    "status": "RUNNING",
                    "pid": os.getpid(),
                    "pending_batch": bi,
                    "completed_batches": completed,
                    "model_seconds": elapsed,
                },
            )
            width = max(len(b["input_ids"]) for b in batch)
            pad = observer.tokenizer.pad_token_id
            if pad is None:
                pad = observer.tokenizer.eos_token_id
            ids = torch.full((8, width), pad, dtype=torch.long, device=cfg["device"])
            mask = torch.zeros_like(ids)
            last = []
            for j, b in enumerate(batch):
                n = len(b["input_ids"])
                ids[j, :n] = torch.tensor(b["input_ids"], device=cfg["device"])
                mask[j, :n] = 1
                last.append(n - 1)
            stamp = time.monotonic()
            try:
                with torch.inference_mode():
                    body = observer.model.model(
                        input_ids=ids,
                        attention_mask=mask,
                        use_cache=False,
                        return_dict=True,
                    )
                    state = body.last_hidden_state[
                        torch.arange(8, device=cfg["device"]),
                        torch.tensor(last, device=cfg["device"]),
                    ]
                    logits = (
                        observer.model.lm_head(state)
                        .float()
                        .cpu()
                        .numpy()
                        .astype(float)
                    )
                    states = state.float().cpu().numpy()
                if not np.isfinite(logits).all() or not np.isfinite(states).all():
                    raise ValueError("Nonfinite observation")
                R.archive(
                    out / "batches" / f"{bi:03d}.npz",
                    logits=logits[:, alpha],
                    normalizer=logsumexp(logits, -1),
                    states=states,
                )
            finally:
                elapsed += time.monotonic() - stamp
            completed += 1
            H.status(
                out,
                {
                    "status": "RUNNING",
                    "pid": os.getpid(),
                    "pending_batch": None,
                    "completed_batches": completed,
                    "model_seconds": elapsed,
                },
            )
            if completed % 12 == 0:
                print(f"HM220 {completed}/48;model_seconds={elapsed:.3f}", flush=True)
            del body, state, logits, states
        H.status(
            out,
            {
                "status": "COLLECTED",
                "pid": None,
                "completed_batches": completed,
                "model_seconds": elapsed,
                "new_API_train_private_calls": 0,
            },
        )
        R.seal(
            out,
            "collected_seal.json",
            ("head.npz", "runtime.json")
            + tuple(f"batches/{i:03d}.npz" for i in range(48)),
        )
        print("COLLECTED HM220", elapsed, flush=True)
    except BaseException as exc:
        H.status(
            out,
            {
                "status": "FAILED",
                "completed_batches": completed,
                "model_seconds": elapsed,
                "error": repr(exc),
                "no_auto_retry": True,
            },
        )
        raise
    finally:
        lock.unlink(missing_ok=True)


def native(out):
    """Read the complete addressed response cube without reading raw texts."""
    rows = [
        H.selected(out / "batches" / f"{i:03d}.npz", ("logits", "normalizer", "states"))
        for i in range(48)
    ]
    return {
        k: np.stack([r[k] for r in rows]).reshape(
            (24, 2, 2, 2, 2) + rows[0][k].shape[1:]
        )
        for k in rows[0]
    }


def stats(values, cfg):
    """Summarize an author-level contrast, with fixed whole-author bootstrap."""
    v = np.asarray(values, float)
    lo, hi = bootstrap(v[:, None], cfg["bootstrap_draws"], cfg["bootstrap_seed"])[0]
    return {
        "mean": float(v.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "positive_authors": int(np.count_nonzero(v > 0)),
    }


def analyze(out):
    """Keep every conditional action and signed character, not selected successes."""
    cfg = R.check(out, ("collected_seal.json",))["config"]
    n = native(out)
    a = M.accounts(n["logits"])
    masks, co, reconstruction = M.characters(a["oriented"])
    mass = np.exp(logsumexp(n["logits"], -1) - n["normalizer"])
    rows = [
        {
            "beneficiary": cfg["beneficiaries"][b],
            "cost": cfg["costs"][c],
            **stats(a["delta"][:, b, c], cfg),
        }
        for b, c in product(range(2), repeat=2)
    ]
    R.archive(
        out / "analysis.npz",
        **a,
        characters=co,
        reconstruction=reconstruction,
        compliance=mass,
    )
    R.table(out / "condition_actions.csv", rows)
    R.table(
        out / "author_actions.csv",
        [
            {
                "author": u,
                "beneficiary": cfg["beneficiaries"][b],
                "cost": cfg["costs"][c],
                "action": float(a["delta"][u, b, c]),
            }
            for u in range(24)
            for b, c in product(range(2), repeat=2)
        ],
    )
    R.table(
        out / "characters.csv",
        [
            {"author": u, "mask": "".join(map(str, m)), "coefficient": float(co[u, k])}
            for u in range(24)
            for k, m in enumerate(masks)
        ],
    )
    R.table(
        out / "endpoint_receipts.csv",
        [
            {
                "author": u,
                "arm": arm,
                "beneficiary": cfg["beneficiaries"][b],
                "cost": cfg["costs"][c],
                "order": o,
                "log_odds": float(a["oriented"][u, arm, b, c, o]),
                "lottery_probability": float(a["probability"][u, arm, b, c, o]),
                "compliance": float(mass[u, arm, b, c, o]),
            }
            for u in range(24)
            for arm, b, c, o in product(range(2), repeat=4)
        ],
    )
    R.write(
        out / "summary.json",
        {
            "packet": "HM220",
            "primary_beneficiary_selective": stats(a["selective"], cfg),
            "cost_interaction": stats(a["cost_interaction"], cfg),
            "conditions": rows,
            "beneficiary_mean": [
                {
                    "beneficiary": cfg["beneficiaries"][b],
                    **stats(a["beneficiary_mean"][:, b], cfg),
                }
                for b in range(2)
            ],
            "compliance_range": [float(mass.min()), float(mass.max())],
            "order_defect_rms": float(np.sqrt(np.mean(a["order_defect"] ** 2))),
            "order_action_defect_rms": float(
                np.sqrt(
                    np.mean((a["order_defect"][:, 0] - a["order_defect"][:, 1]) ** 2)
                )
            ),
            "character_reconstruction_error": float(
                np.max(abs(reconstruction - a["oriented"]))
            ),
            "new_batches": 48,
            "new_endpoints": 384,
            "actual_authors": 24,
            "original_scope_complete": False,
            "human_behavior_or_personality_accuracy": False,
        },
    )
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "analysis.npz",
            "condition_actions.csv",
            "author_actions.csv",
            "characters.csv",
            "endpoint_receipts.csv",
            "summary.json",
        ),
    )
    print(
        "ANALYZED HM220",
        R.read(out / "summary.json")["primary_beneficiary_selective"],
        flush=True,
    )


def verify(out):
    """Replay odds, all factorial accounts, primary CI and head via direct sums."""
    cfg = R.check(out, ("collected_seal.json", "analyzed_seal.json"))["config"]
    n = native(out)
    saved = H.selected(
        out / "analysis.npz",
        (
            "oriented",
            "delta",
            "selective",
            "cost_interaction",
            "characters",
            "compliance",
        ),
    )
    direct = np.log(np.exp(n["logits"][..., 0]) / np.exp(n["logits"][..., 1]))
    direct[..., 1] *= -1
    error = float(np.max(abs(direct - saved["oriented"])))
    delta = np.empty((24, 2, 2))
    selective, interaction = [], []
    for u in range(24):
        for b, c in product(range(2), repeat=2):
            delta[u, b, c] = sum(
                (direct[u, 0, b, c, o] - direct[u, 1, b, c, o]) / 2 for o in range(2)
            )
        selective.append(sum(delta[u, 0, c] - delta[u, 1, c] for c in range(2)) / 2)
        interaction.append(
            delta[u, 0, 1] - delta[u, 0, 0] - delta[u, 1, 1] + delta[u, 1, 0]
        )
    for a, b in (
        (delta, saved["delta"]),
        (selective, saved["selective"]),
        (interaction, saved["cost_interaction"]),
    ):
        error = max(error, float(np.max(abs(np.asarray(a) - b))))
    for u in range(24):
        for k, m in enumerate(product((0, 1), repeat=4)):
            value = sum(
                direct[(u,) + cell]
                * np.prod([M.SIGNS[d, cell[d]] for d in range(4) if m[d]])
                / 16
                for cell in product((0, 1), repeat=4)
            )
            error = max(error, abs(value - saved["characters"][u, k]))
    summary = R.read(out / "summary.json")
    direct_mass = np.exp(n["logits"][..., 0] - n["normalizer"]) + np.exp(
        n["logits"][..., 1] - n["normalizer"]
    )
    error = max(error, float(np.max(abs(direct_mass - saved["compliance"]))))
    error = max(
        error,
        float(
            np.max(
                abs(
                    np.asarray(summary["compliance_range"])
                    - np.asarray([direct_mass.min(), direct_mass.max()])
                )
            )
        ),
    )
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    draws = np.asarray(selective)[
        rng.integers(0, 24, size=(cfg["bootstrap_draws"], 24))
    ].mean(-1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    primary = summary["primary_beneficiary_selective"]
    error = max(
        error,
        abs(primary["mean"] - np.mean(selective)),
        abs(primary["ci_low"] - lo),
        abs(primary["ci_high"] - hi),
    )
    if primary["positive_authors"] != np.count_nonzero(np.asarray(selective) > 0):
        raise ValueError("Primary direction count mismatch")
    weights = H.selected(out / "head.npz", ("weights",))["weights"].astype(float)
    head_error = float(np.max(abs(n["states"].astype(float) @ weights.T - n["logits"])))
    if error > 1e-10 or head_error > 2e-4:
        raise ValueError(f"Observation/account error: {error}/{head_error}")
    R.write(
        out / "verification.json",
        {
            "PASS": True,
            "max_alternate_error": float(error),
            "native_head_max_error": head_error,
            "literal_endpoint_accounts": 384,
            "literal_character_accounts": 384,
            "literal_compliance_accounts": 384,
            "same_agent_not_blind": True,
        },
    )
    R.seal(out, "verification_seal.json", ("verification.json",))
    print("VERIFIED HM220", error, head_error, flush=True)


def main():
    """Run exactly one explicitly selected stage; no queue or automatic retry."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("prepare", "collect", "analyze", "verify", "check")
    )
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        R.check(
            args.out,
            ("collected_seal.json", "analyzed_seal.json", "verification_seal.json"),
        )
        print("ALL HM220 SEALS PASS", flush=True)
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
