#!/usr/bin/env python3
"""Finite name/definition intervention on the sealed SOURCE-selected evidence."""

import argparse
import hashlib
import os
import sys
import time
from collections.abc import Mapping
from itertools import combinations, product
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_target_contract as M
from grain_research.construction.math_prospective_evidence import bootstrap
from grain_research.vector_mechanisms.navigation import MODEL
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_prospective_evidence_hm218 as H
from scripts.run_grain_vector_mechanism_external_features import bundle_arms

CONFIG = "configs/grain_target_contract_hm221.json"
DEFAULT = ROOT / "results/grain_math_response_v1/target_contract_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_TARGET_CONTRACT_PLAN_HM221.md",
    "grain_research/construction/math_target_contract.py",
    "scripts/run_grain_target_contract_hm221.py",
    "tests/test_grain_target_contract.py",
)


def prepare(out):
    """Bind old actual events and freeze all contracts/tokens before responses."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve every existing or uncertain run")
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["parent"]
    stages = ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")
    pm = R.check(parent, stages)
    previous = ROOT / pm["config"]["parent"]
    gm = R.check(previous, stages)
    material = ROOT / gm["config"]["material"] / "private_material.json"
    bundles = R.read(material)["bundles"]
    panel = H.selected(parent / "observations.npz", ("panel_indices",))["panel_indices"]
    blocks, selections = (
        R.read(parent / "blocks.json"),
        R.read(previous / "selections.json"),
    )
    source = H.selected(parent / "observations.npz", ("source_actions",))[
        "source_actions"
    ][:, 0, 2]
    tok = AutoTokenizer.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False
    )
    codes = [tok.encode(str(k), add_special_tokens=False) for k in range(10)]
    if any(len(row) != 1 for row in codes) or len({row[0] for row in codes}) != 10:
        raise ValueError("Distinct single-token digit alphabet required")
    units, bindings, lengths = [], [], []

    def add(author, arm, texts):
        for n, d in product(range(3), range(2)):
            user = M.render(texts, n, d)
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
                raise ValueError("Actual chat token IDs required")
            if not ids or len(ids) + 16 > cfg["max_tokens"]:
                raise ValueError("Unsupported input: no truncation or author omission")
            units.append(
                {
                    "author": author,
                    "arm": arm,
                    "name": n,
                    "definition": d,
                    "input_ids": ids,
                }
            )
            lengths.append(len(ids))

    for u, row in enumerate(blocks):
        if row["author"] != u:
            raise ValueError("Changed author address")
        bundle = bundles[int(panel[u])]
        events = [e["comment_id"] for e in bundle["views"][2]]
        if events != selections[u]["event_ids"]:
            raise ValueError("Changed event identities")
        texts = bundle_arms(bundle)["cue_mask"][24:36]
        for arm, label in enumerate(("support", "counter")):
            slots = sorted(row[label])
            if slots != sorted(selections[u][label]):
                raise ValueError("Changed SOURCE selection")
            selected = [texts[j] for j in slots]
            bindings.append(
                {
                    "author": u,
                    "arm": arm,
                    "event_ids": [events[j] for j in slots],
                    "body_hashes": [
                        hashlib.sha256(t.encode()).hexdigest() for t in selected
                    ],
                }
            )
            add(u, arm, selected)
    add(-1, -1, [])
    if len(blocks) != cfg["authors"] or len(units) != cfg["enquiries"]:
        raise ValueError("Incomplete registered scope")
    files = {
        p.name: R.digest(p)
        for p in MODEL.iterdir()
        if p.is_file() and not p.name.startswith("._")
    }
    if files != pm["model_files"]:
        raise ValueError("Changed model")
    (out / "batches").mkdir(parents=True, exist_ok=False)
    R.write(out / "units.json", units)
    R.write(out / "bindings.json", bindings)
    R.write(
        out / "contracts.json",
        [M.contract(n, d) for n, d in product(range(3), range(2))],
    )
    R.archive(
        out / "source_prediction.npz",
        departure=source,
        primary=source[:, 2] - source[:, 4],
    )
    R.write(
        out / "input_quality.json",
        {
            "authors": 24,
            "reused_event_addresses": 288,
            "new_endpoints": len(units),
            "prompt_tokens_range": [min(lengths), max(lengths)],
            "no_reselection_or_input_truncation": True,
            "all_contracts_before_native_answers": True,
        },
    )
    inputs = [
        parent / k
        for k in (
            "manifest.json",
            "blocks.json",
            "observations.npz",
            "verification_seal.json",
        )
    ]
    inputs += [previous / "manifest.json", previous / "selections.json", material]
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "alphabet": {
                "labels": list(map(str, range(10))),
                "token_ids": [r[0] for r in codes],
            },
            "model_files": files,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "old_results_known_new_contract_actions_unobserved": True,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        (
            "units.json",
            "bindings.json",
            "contracts.json",
            "source_prediction.npz",
            "input_quality.json",
        ),
    )
    H.status(out, {"status": "PREPARED", "completed_batches": 0})
    print("PREPARED HM221:24authors/49batches/294endpoints", flush=True)


def collect(out):
    """Observe every fixed endpoint once with bounded cost, no response fitting."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    manifest = R.check(out)
    cfg = manifest["config"]
    if R.read(out / "runtime.json")["status"] != "PREPARED":
        raise ValueError("No uncertain/terminal automatic retry")
    for name, digest in manifest["model_files"].items():
        if R.digest(MODEL / name) != digest:
            raise ValueError("Observer changed")
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
        size = cfg["batch_size"]
        for start in range(0, len(units), size):
            batch, bi = units[start : start + size], start // size
            if (
                len(batch) != size
                or len({(b["author"], b["arm"]) for b in batch}) != 1
                or [(b["name"], b["definition"]) for b in batch]
                != list(product(range(3), range(2)))
            ):
                raise ValueError("Changed endpoint binding")
            if elapsed >= cfg["max_model_seconds"]:
                raise RuntimeError("Finite cap reached: preserve, no extension")
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
            ids = torch.full((size, width), pad, dtype=torch.long, device=cfg["device"])
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
                        torch.arange(size, device=cfg["device"]),
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
                    raise ValueError("Nonfinite observations")
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
            if completed % 12 == 0 or completed == cfg["batches"]:
                print(f"HM221 {completed}/49;model_seconds={elapsed:.3f}", flush=True)
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
            + tuple(f"batches/{i:03d}.npz" for i in range(cfg["batches"])),
        )
        print("COLLECTED HM221", elapsed, flush=True)
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
    """Load the complete addressed field and separate the empty control."""
    rows = [
        H.selected(out / "batches" / f"{i:03d}.npz", ("logits", "normalizer", "states"))
        for i in range(49)
    ]
    data = {
        k: np.stack([r[k] for r in rows[:48]]).reshape(
            (24, 2, 3, 2) + rows[0][k].shape[1:]
        )
        for k in rows[0]
    }
    for key, value in rows[-1].items():
        data["empty_" + key] = value.reshape((3, 2) + value.shape[1:])
    return data


def stats(values, cfg):
    """Report a fixed author contrast and its whole-author conditional interval."""
    v = np.asarray(values, float)
    lo, hi = bootstrap(v[:, None], cfg["bootstrap_draws"], cfg["bootstrap_seed"])[0]
    return {
        "mean": float(v.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "positive_authors": int(np.count_nonzero(v > 0)),
    }


def analyze(out):
    """Keep all name, criterion, interaction, vector and nonlinear accounts."""
    cfg = R.check(out, ("collected_seal.json",))["config"]
    n = native(out)
    y, linear, curvature = M.responses(n["logits"])
    a, la, ca = (M.action_accounts(v) for v in (y, linear, curvature))
    va = M.action_accounts(n["states"])
    empty = M.responses(n["empty_logits"])[0]
    mass = np.exp(logsumexp(n["logits"], -1) - n["normalizer"])
    empty_mass = np.exp(logsumexp(n["empty_logits"], -1) - n["empty_normalizer"])
    rows = [
        {
            "name": cfg["names"][i],
            "definition": cfg["definitions"][j],
            **stats(a["delta"][:, i, j], cfg),
        }
        for i, j in product(range(3), range(2))
    ]
    contrasts = [
        {
            "contrast": "definition_at_" + cfg["names"][i],
            **stats(a["definition_by_name"][:, i], cfg),
        }
        for i in range(3)
    ]
    contrasts += [
        {"contrast": k, **stats(a[k], cfg)}
        for k in ("named_definition", "named_name", "interaction_corner")
    ]
    geometry = []
    for u in range(24):
        for i, j in combinations(range(3), 2):
            x, z = va["definition_by_name"][u, i], va["definition_by_name"][u, j]
            dot = float(x @ z)
            norm = float(np.linalg.norm(x) * np.linalg.norm(z))
            geometry.append(
                {
                    "author": u,
                    "name0": cfg["names"][i],
                    "name1": cfg["names"][j],
                    "inner_product": dot,
                    "cosine": dot / norm if norm else None,
                }
            )
    R.archive(
        out / "analysis.npz",
        readings=y,
        linear_readings=linear,
        curvature_readings=curvature,
        compliance=mass,
        empty_readings=empty,
        empty_compliance=empty_mass,
        **a,
        vector_delta=va["delta"],
        vector_coefficients=va["coefficients"],
        vector_definition_by_name=va["definition_by_name"],
        linear_delta=la["delta"],
        curvature_delta=ca["delta"],
    )
    R.table(out / "condition_actions.csv", rows)
    R.table(out / "contrasts.csv", contrasts)
    R.table(out / "vector_geometry.csv", geometry)
    R.table(
        out / "author_actions.csv",
        [
            {
                "author": u,
                "name": cfg["names"][i],
                "definition": cfg["definitions"][j],
                "action": float(a["delta"][u, i, j]),
                "linear": float(la["delta"][u, i, j]),
                "curvature": float(ca["delta"][u, i, j]),
            }
            for u in range(24)
            for i, j in product(range(3), range(2))
        ],
    )
    source = H.selected(out / "source_prediction.npz", ("departure", "primary"))
    R.write(
        out / "summary.json",
        {
            "packet": "HM221",
            "primary_anonymous_definition": stats(a["anonymous_definition"], cfg),
            "all_contrasts": contrasts,
            "conditions": rows,
            "source_criteria_not_identical_to_Big5": True,
            "source_departure_A_O_means": [
                float(source["departure"][:, q].mean()) for q in (2, 4)
            ],
            "source_A_minus_O_mean": float(source["primary"].mean()),
            "source_native_anonymous_sign_agreement": int(
                np.count_nonzero(source["primary"] * a["anonymous_definition"] > 0)
            ),
            "named_reconstruction_error": float(
                np.max(abs(a["delta"][:, :2] - a["reconstruction"]))
            ),
            "response_linear_curvature_error": float(
                np.max(abs(y - linear - curvature))
            ),
            "empty_readings": empty.tolist(),
            "compliance_range": [
                float(min(mass.min(), empty_mass.min())),
                float(max(mass.max(), empty_mass.max())),
            ],
            "new_batches": 49,
            "new_endpoints": 294,
            "actual_authors": 24,
            "human_behavior_or_personality_accuracy": False,
            "original_scope_complete": False,
        },
    )
    files = (
        "analysis.npz",
        "condition_actions.csv",
        "contrasts.csv",
        "vector_geometry.csv",
        "author_actions.csv",
        "summary.json",
    )
    R.seal(out, "analyzed_seal.json", files)
    print(
        "ANALYZED HM221",
        R.read(out / "summary.json")["primary_anonymous_definition"],
        flush=True,
    )


def verify(out):
    """Check the same frozen observations by literal sums and contractions."""
    cfg = R.check(out, ("collected_seal.json", "analyzed_seal.json"))["config"]
    n = native(out)
    saved = H.selected(
        out / "analysis.npz",
        (
            "readings",
            "delta",
            "coefficients",
            "anonymous_definition",
            "compliance",
            "vector_coefficients",
            "linear_readings",
            "curvature_readings",
        ),
    )
    z = n["logits"]
    shifted = z - z.max(-1, keepdims=True)
    direct = np.log(np.exp(shifted[..., 5:]).sum(-1) / np.exp(shifted[..., :5]).sum(-1))
    delta = direct[:, 0] - direct[:, 1]
    error = max(
        float(np.max(abs(direct - saved["readings"]))),
        float(np.max(abs(delta - saved["delta"]))),
    )
    for u in range(24):
        for k, (i, j) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
            co = sum(
                delta[u, n0, d] * (-1) ** (n0 * i + d * j) / 4
                for n0, d in product(range(2), repeat=2)
            )
            error = max(error, abs(co - saved["coefficients"][u, k]))
    primary = delta[:, 2, 0] - delta[:, 2, 1]
    error = max(error, float(np.max(abs(primary - saved["anonymous_definition"]))))
    mass = sum(np.exp(z[..., j] - n["normalizer"]) for j in range(10))
    error = max(error, float(np.max(abs(mass - saved["compliance"]))))
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    draws = primary[rng.integers(0, 24, size=(cfg["bootstrap_draws"], 24))].mean(-1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    s = R.read(out / "summary.json")["primary_anonymous_definition"]
    error = max(
        error,
        abs(s["mean"] - primary.mean()),
        abs(s["ci_low"] - lo),
        abs(s["ci_high"] - hi),
    )
    if s["positive_authors"] != np.count_nonzero(primary > 0):
        raise ValueError("Changed primary sign count")
    weights = H.selected(out / "head.npz", ("weights",))["weights"].astype(float)
    w = weights[5:].mean(0) - weights[:5].mean(0)
    head_error = max(
        float(np.max(abs(n["states"].astype(float) @ weights.T - z))),
        float(
            np.max(abs(n["empty_states"].astype(float) @ weights.T - n["empty_logits"]))
        ),
    )
    contraction = float(
        np.max(
            abs(
                saved["vector_coefficients"] @ w
                - M.action_accounts(saved["linear_readings"])["coefficients"]
            )
        )
    )
    error = max(
        error,
        float(
            np.max(abs(direct - saved["linear_readings"] - saved["curvature_readings"]))
        ),
    )
    if error > 1e-10 or max(head_error, contraction) > 2e-4:
        raise ValueError(
            f"Observation/account mismatch: {error}/{head_error}/{contraction}"
        )
    R.write(
        out / "verification.json",
        {
            "PASS": True,
            "max_alternate_error": error,
            "native_head_max_error": head_error,
            "vector_contraction_error": contraction,
            "literal_digit_odds": 288,
            "literal_named_coefficients": 96,
            "literal_probability_masses": 288,
            "same_agent_not_blind": True,
        },
    )
    R.seal(out, "verification_seal.json", ("verification.json",))
    print("VERIFIED HM221", error, head_error, contraction, flush=True)


def main():
    """Execute one explicit finite stage without automatic retries or queues."""
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
        print("ALL HM221 SEALS PASS", flush=True)
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
