#!/usr/bin/env python3
"""Frozen real evidence-alone operation following the prospective HM218 rule."""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_evidence_background as M
from grain_research.construction import math_prospective_evidence as P
from grain_research.vector_mechanisms import direct_traits as D
from grain_research.vector_mechanisms.navigation import MODEL
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_prospective_evidence_hm218 as H
from scripts.run_grain_vector_mechanism_external_features import bundle_arms

CONFIG = "configs/grain_evidence_background_hm219.json"
DEFAULT = ROOT / "results/grain_math_response_v1/evidence_background_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_EVIDENCE_BACKGROUND_PLAN_HM219.md",
    "grain_research/construction/math_evidence_background.py",
    "scripts/run_grain_evidence_background_hm219.py",
    "tests/test_grain_evidence_background.py",
    "scripts/run_grain_prospective_evidence_hm218.py",
    "grain_research/construction/math_prospective_evidence.py",
    "grain_research/vector_mechanisms/direct_traits.py",
)
STAGES = ("alone", "with_background", "background_interaction")


def prepare(out):
    """Bind actual S/T/R blocks and unchanged old tokens before new answers."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve sealed/uncertain outputs; no overwrite")
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["parent"]
    pm = R.check(
        parent, ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    bank = H.selected(
        parent / "observations.npz",
        ("source_means", "codes", "variance", "vectors", "panel_indices", "accounts"),
    )
    choices = R.read(parent / "selections.json")
    old_units = R.read(parent / "units.json")
    material = ROOT / pm["config"]["material"]
    bundles = R.read(material / "private_material.json")["bundles"]
    tok = AutoTokenizer.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False
    )
    tasks = tuple(D.BIG5)
    units, blocks, source_actions, lengths = [], [], [], []
    for u, row in enumerate(choices):
        s, t = tuple(row["support"]), tuple(row["counter"])
        r = M.partition(s, t)
        bundle = bundles[int(bank["panel_indices"][u])]
        if [e["comment_id"] for e in bundle["views"][2]] != row["event_ids"]:
            raise ValueError("Changed actual event addresses")
        texts = bundle_arms(bundle)["cue_mask"][24:36]
        for arm, excluded in ((1, s), (2, t)):
            keep = [j for j in range(12) if j not in excluded]
            for q, task in enumerate(tasks):
                ids, _, _ = D.tokenize_prompt([texts[j] for j in keep], task, tok)
                inherited = old_units[(u * 3 + arm) * 5 + q]
                if ids != inherited["input_ids"]:
                    raise ValueError("Inherited native observation changed")
        for arm, driver in enumerate((s, t)):
            for q, task in enumerate(tasks):
                ids, _, _ = D.tokenize_prompt(
                    [texts[j] for j in sorted(driver)], task, tok
                )
                if len(ids) + 16 > cfg["max_tokens"]:
                    raise ValueError("Unsupported material; no truncation")
                units.append({"author": u, "arm": arm, "target": q, "input_ids": ids})
                lengths.append(len(ids))
        values = P.accounts(
            bank,
            np.stack([bank["vectors"][u, list(d)].mean(0) for d in (s, t)]),
        )
        alone = values[0] - values[1]
        with_r = (
            bank["accounts"][u, row["counter_index"]]
            - bank["accounts"][u, row["support_index"]]
        )
        source_actions.append(np.stack((alone, with_r, with_r - alone)))
        blocks.append({"author": u, "support": s, "counter": t, "background": r})
    if len(choices) != cfg["authors"] or len(units) != cfg["enquiries"]:
        raise ValueError("Incomplete declared observation scope")
    model_files = {
        p.name: R.digest(p)
        for p in MODEL.iterdir()
        if p.is_file() and not p.name.startswith("._")
    }
    if model_files != pm["model_files"]:
        raise ValueError("Changed frozen native model")
    (out / "batches").mkdir(parents=True, exist_ok=False)
    R.archive(
        out / "observations.npz", source_actions=np.asarray(source_actions), **bank
    )
    R.write(out / "units.json", units)
    R.write(out / "blocks.json", blocks)
    R.write(
        out / "input_quality.json",
        {
            "authors": 24,
            "events_reused": 288,
            "new_batches": 48,
            "new_enquiries": 240,
            "inherited_8comment_token_comparisons": 240,
            "inherited_token_difference": 0,
            "prompt_tokens_range": [min(lengths), max(lengths)],
            "new_selections_or_application_labels": False,
            "new_raw_preprocessing": False,
            "all_predictions_before_new_native": True,
        },
    )
    inputs = [
        parent / name
        for name in (
            "manifest.json",
            "observations.npz",
            "selections.json",
            "units.json",
            "collected_seal.json",
            "analyzed_seal.json",
            "verification_seal.json",
        )
    ] + [material / "private_material.json"]
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "alphabet": pm["alphabet"],
            "model_files": model_files,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "old_results_known_new_answers_not_observed": True,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        (
            "observations.npz",
            "units.json",
            "blocks.json",
            "input_quality.json",
        ),
    )
    H.status(out, {"status": "PREPARED", "completed_batches": 0})
    print("PREPARED HM219:24authors/48new batches/240endpoints", flush=True)


def collect(out):
    """Read the finite driver family once, with recoverable failure receipts."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    manifest = R.check(out)
    cfg = manifest["config"]
    if R.read(out / "runtime.json")["status"] != "PREPARED":
        raise ValueError("No automatic uncertain or terminal retry")
    for name, h in manifest["model_files"].items():
        if R.digest(MODEL / name) != h:
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
        for start in range(0, len(units), 5):
            batch, bi = units[start : start + 5], start // 5
            if [b["target"] for b in batch] != list(range(5)) or len(
                {(b["author"], b["arm"]) for b in batch}
            ) != 1:
                raise ValueError("Endpoint addressing changed")
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
            ids = torch.full((5, width), pad, dtype=torch.long, device=cfg["device"])
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
                        torch.arange(5, device=cfg["device"]),
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
                print(
                    f"HM219 observed {completed}/48;model_seconds={elapsed:.3f}",
                    flush=True,
                )
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
        print("COLLECTED HM219", elapsed, flush=True)
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
    """Load exactly the two declared driver arms for every author and target."""
    rows = [
        H.selected(out / "batches" / f"{i:03d}.npz", ("logits", "normalizer", "states"))
        for i in range(48)
    ]
    return {
        k: np.stack([r[k] for r in rows]).reshape((24, 2) + rows[0][k].shape)
        for k in rows[0]
    }


def derived(out):
    """Join sealed SAME-event parent responses to new driver responses, not fit."""
    cfg = R.check(out, ("collected_seal.json",))["config"]
    parent = ROOT / cfg["parent"]
    R.check(
        parent, ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    n = native(out)
    old = H.selected(parent / "analysis.npz", ("raw",))["raw"]
    y = M.readings(n["logits"])
    a = M.account(old, y, cfg["evaluation_direction"])
    return cfg, n, y, a


def analyze(out):
    """Report all raw/common/specific context actions and predeclared primary."""
    cfg, n, y, a = derived(out)
    z = R.prepared(out)
    rows, stage_rows = [], []
    for k, stage in enumerate(STAGES):
        ci = {
            name: P.bootstrap(
                a[name][:, k], cfg["bootstrap_draws"], cfg["bootstrap_seed"]
            )
            for name in ("raw", "specific", "oriented_remainder")
        }
        for q, target in enumerate(cfg["targets"]):
            row = {"stage": stage, "target": target}
            for name in ci:
                row.update(
                    {
                        name + "_mean": float(a[name][:, k, q].mean()),
                        name + "_ci_low": float(ci[name][q, 0]),
                        name + "_ci_high": float(ci[name][q, 1]),
                        name + "_positive": int(np.count_nonzero(a[name][:, k, q] > 0)),
                    }
                )
            row["source_departure_mean"] = float(z["source_actions"][:, k, 2, q].mean())
            rows.append(row)
        for name in ("common", "oriented_common"):
            v = a[name][:, k, None]
            lo, hi = P.bootstrap(v, cfg["bootstrap_draws"], cfg["bootstrap_seed"])[0]
            stage_rows.append(
                {
                    "stage": stage,
                    "account": name,
                    "mean": float(v.mean()),
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                }
            )
    mass = np.exp(logsumexp(n["logits"], -1) - n["normalizer"])
    R.archive(out / "analysis.npz", readings=y, compliance=mass, **a)
    R.table(out / "actions_by_target.csv", rows)
    R.table(out / "common_accounts.csv", stage_rows)
    R.table(
        out / "actions_by_author.csv",
        [
            {
                "author": u,
                "stage": stage,
                "target": cfg["targets"][q],
                "raw": float(a["raw"][u, k, q]),
                "specific": float(a["specific"][u, k, q]),
                "oriented_remainder": float(a["oriented_remainder"][u, k, q]),
                "source_departure": float(z["source_actions"][u, k, 2, q]),
            }
            for u in range(24)
            for k, stage in enumerate(STAGES)
            for q in range(5)
        ],
    )
    R.write(
        out / "summary.json",
        {
            "packet": "HM219",
            "primary": rows[12],
            "all_targets": rows,
            "common_accounts": stage_rows,
            "compliance_range": [float(mass.min()), float(mass.max())],
            "new_batches": 48,
            "new_enquiries": 240,
            "reused_batches": 72,
            "new_authors_API_fit_training": 0,
            "no_personality_accuracy_claim": True,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "analysis.npz",
            "actions_by_target.csv",
            "common_accounts.csv",
            "actions_by_author.csv",
            "summary.json",
        ),
    )
    print("ANALYZED HM219", rows[12], flush=True)


def verify(out):
    """Use literal kernels, direct odds and mixed differences for alternate audit."""
    R.check(out, ("collected_seal.json", "analyzed_seal.json"))
    cfg, n, y, a = derived(out)
    z = R.prepared(out)
    blocks = R.read(out / "blocks.json")
    manual_y = np.log(
        np.exp(n["logits"][..., 5:]).sum(-1) / np.exp(n["logits"][..., :5]).sum(-1)
    )
    old = H.selected(ROOT / cfg["parent"] / "analysis.npz", ("native_readings",))[
        "native_readings"
    ]
    direct = np.stack(
        (
            manual_y[:, 0] - manual_y[:, 1],
            old[:, 2] - old[:, 1],
            (old[:, 2] - manual_y[:, 0]) - (old[:, 1] - manual_y[:, 1]),
        ),
        1,
    )
    errors = [float(np.max(abs(y - manual_y))), float(np.max(abs(direct - a["raw"])))]
    direction = np.asarray(cfg["evaluation_direction"])
    errors.append(
        float(
            np.max(
                abs(
                    a["oriented_common"][..., None] * direction
                    + a["oriented_remainder"]
                    - direct
                )
            )
        )
    )
    primary = R.read(out / "summary.json")["primary"]
    specific = direct[:, 2] - direct[:, 2].mean(-1, keepdims=True)
    ix = np.random.default_rng(cfg["bootstrap_seed"]).integers(
        24, size=(cfg["bootstrap_draws"], 24)
    )
    lo, hi = np.quantile(specific[ix, 2].mean(1), [0.025, 0.975])
    errors.extend(
        (
            abs(float(specific[:, 2].mean()) - primary["specific_mean"]),
            abs(float(lo) - primary["specific_ci_low"]),
            abs(float(hi) - primary["specific_ci_high"]),
        )
    )
    if int(np.count_nonzero(specific[:, 2] > 0)) != primary["specific_positive"]:
        raise ValueError("Primary author count disagrees with literal interaction")
    stored = H.selected(out / "analysis.npz", tuple(a) + ("readings",))
    errors.extend(float(np.max(abs(stored[name] - value))) for name, value in a.items())
    source_count = 0
    codes, variance = z["codes"], float(z["variance"])
    for u, row in enumerate(blocks):
        if M.partition(row["support"], row["counter"]) != tuple(row["background"]):
            raise ValueError("Changed block partition")
        for k, include_r in enumerate((False, True)):
            vals = []
            for name in ("support", "counter"):
                slots = row[name] + (row["background"] if include_r else [])
                point = z["vectors"][u, slots].mean(0)
                kernel = np.exp(
                    -np.sum((z["source_means"] - point) ** 2, -1) / (2 * variance)
                ).mean(1)
                law = np.array(
                    [kernel[codes == g].sum() / len(codes) for g in range(32)]
                )
                account = []
                for q in range(5):
                    high = (np.arange(32) >> q) & 1 == 1
                    c = codes & ~(1 << q)
                    base = np.array(
                        [
                            kernel[c == (g & ~(1 << q))].mean()
                            * np.count_nonzero(codes == g)
                            / len(codes)
                            for g in range(32)
                        ]
                    )
                    f, b = (
                        np.log(law[high].sum() / law[~high].sum()),
                        np.log(base[high].sum() / base[~high].sum()),
                    )
                    account.append([f, b, f - b])
                vals.append(np.asarray(account).T)
            errors.append(
                float(np.max(abs((vals[0] - vals[1]) - z["source_actions"][u, k])))
            )
            source_count += 15
    weights = H.selected(out / "head.npz", ("weights",))["weights"].astype(float)
    head_error = float(np.max(abs(n["states"].astype(float) @ weights.T - n["logits"])))
    error = max(errors)
    if error > 1e-11 or head_error > 3e-4:
        raise ValueError("Registered arithmetic audit failed; no rescue")
    R.write(
        out / "verification.json",
        {
            "PASS": True,
            "max_algebra_source_error": error,
            "literal_source_action_accounts": source_count,
            "native_head_max_error": head_error,
            "new_native_endpoints": 240,
            "same_agent_not_blind": True,
        },
    )
    R.seal(out, "verification_seal.json", ("verification.json",))
    print("VERIFIED HM219", error, head_error, flush=True)


def main():
    """Execute one explicit stage; never automatically retry or launch queues."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage", choices=("prepare", "collect", "analyze", "verify", "check")
    )
    p.add_argument("--output", type=Path, default=DEFAULT)
    args = p.parse_args()
    if args.stage == "check":
        R.check(
            args.output,
            ("collected_seal.json", "analyzed_seal.json", "verification_seal.json"),
        )
        print("ALL HM219 SEALS PASS", flush=True)
    else:
        globals()[args.stage](args.output)


if __name__ == "__main__":
    main()
