#!/usr/bin/env python3
"""Observe one absolute author task under three fixed background contracts."""

import argparse
import hashlib
import os
import shutil
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_absolute_context_observation as M
from grain_research.construction import math_full_attention_rows as F
from grain_research.vector_mechanisms import direct_traits as D
from grain_research.vector_mechanisms.navigation import MODEL
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_prospective_evidence_hm218 as H
from scripts.run_grain_vector_mechanism_external_features import bundle_arms

CONFIG = "configs/grain_absolute_context_observation_hm230.json"
DEFAULT = ROOT / "results/grain_math_response_v1/absolute_context_observation_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_ABSOLUTE_CONTEXT_OBSERVATION_PLAN_HM230.md",
    "grain_research/construction/math_absolute_context_observation.py",
    "scripts/run_grain_absolute_context_observation_hm230.py",
    "tests/test_grain_absolute_context_observation.py",
    "grain_research/construction/math_full_attention_rows.py",
    "grain_research/vector_mechanisms/direct_traits.py",
    "grain_research/vector_mechanisms/trace.py",
    "scripts/run_grain_observed_family_hm170.py",
    "scripts/run_grain_prospective_evidence_hm218.py",
    "scripts/run_grain_vector_mechanism_external_features.py",
)
STAGES = ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")


def finite(value):
    """Represent an undefined contact as missing, not zero evidence."""
    return float(value) if np.isfinite(value) else None


def prepare(out):
    """Bind complete public histories, event occurrences and original task tokens."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("No replacement of uncertain or completed science")
    cfg = R.read(ROOT / CONFIG)
    parent, law, comparison = (
        ROOT / cfg[k] for k in ("parent", "source_law", "comparison")
    )
    pm = R.check(parent, STAGES)
    R.check(law, ("computed_seal.json", "analyzed_seal.json", "verification_seal.json"))
    R.check(comparison, STAGES + ("ordinal_seal.json",))
    material = ROOT / cfg["material"]
    if R.digest(material) != pm["inputs"][cfg["material"]]:
        raise ValueError("Changed authorized public material")
    obs = H.selected(parent / "observations.npz", ("panel_indices", "vectors"))
    selections, old_units = (
        R.read(parent / "selections.json"),
        R.read(parent / "units.json"),
    )
    bundles = R.read(material)["bundles"]
    texts, bindings = [], []
    for u, original in enumerate(obs["panel_indices"]):
        b = bundles[int(original)]
        event_ids = [e["comment_id"] for e in b["views"][2]]
        if (
            b["source_group"] != "big5_only"
            or event_ids != selections[u]["event_ids"]
            or len(set(event_ids)) != 12
        ):
            raise ValueError("Changed PUBLIC twelve-event author binding")
        body = bundle_arms(b)["cue_mask"][24:36]
        texts.append(body)
        bindings.append(
            {
                "author": u,
                "original_author_index": int(original),
                "target_event_ids": event_ids,
                "target_body_hashes": [
                    hashlib.sha256(t.encode()).hexdigest() for t in body
                ],
                "partner": int(M.partners()[u]),
            }
        )
    if len(texts) != cfg["authors"] or obs["vectors"].shape != (24, 12, 1024):
        raise ValueError("Full original24-author scope required")
    tok = AutoTokenizer.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False
    )
    alphabet = D.candidates("agreeableness", tok)
    if alphabet != pm["alphabet"]:
        raise ValueError("The original digit alphabet must remain unchanged")
    units, occurrences, quality = [], [], []
    for u in range(24):
        for mode, name in enumerate(M.MODES):
            owner = None if mode == 0 else u if mode == 1 else int(M.partners()[u])
            background = None if owner is None else texts[owner]
            for q, task in enumerate(D.BIG5):
                user = M.render(texts[u], task, background)
                ids = tok.apply_chat_template(
                    [
                        {"role": "system", "content": D.SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                if isinstance(ids, Mapping):
                    ids = ids["input_ids"]
                if (
                    not ids
                    or any(not isinstance(i, int) or i < 0 for i in ids)
                    or len(ids) + 16 > cfg["max_tokens"]
                ):
                    raise ValueError(
                        "Unsupported complete history; no truncation or shrink"
                    )
                if mode == 0 and ids != old_units[u * 15 + q]["input_ids"]:
                    raise ValueError("Alone prompt is not the original target task")
                units.append({"author": u, "mode": mode, "target": q, "input_ids": ids})
                quality.append(
                    {
                        "author": u,
                        "mode": name,
                        "target": cfg["targets"][q],
                        "tokens": len(ids),
                    }
                )
            occurrences.append(
                {
                    "author": u,
                    "mode": name,
                    "target_author": u,
                    "target_event_ids": bindings[u]["target_event_ids"],
                    "background_author": owner,
                    "background_event_ids": []
                    if owner is None
                    else bindings[owner]["target_event_ids"],
                    "background_body_hashes": []
                    if owner is None
                    else bindings[owner]["target_body_hashes"],
                    "distinct_target_human_events": 12,
                    "duplicate_is_not_new_evidence": mode == 1,
                }
            )
    if len(units) != cfg["enquiries"] or cfg["modes"] != list(M.MODES):
        raise ValueError("Changed registered observation scope")
    model_files = {
        p.name: R.digest(p)
        for p in MODEL.iterdir()
        if p.is_file() and not p.name.startswith("._")
    }
    if model_files != pm["model_files"]:
        raise ValueError("Changed fixed observer model")
    source_records = R.read(law / "cold_scores.json")
    source_means = np.asarray([r["mean"][0] for r in source_records], float)
    if [r["author"] for r in source_records] != list(
        range(24)
    ) or source_means.shape != (24, 5):
        raise ValueError("Changed full-history SOURCE-law address")
    inputs = [material]
    inputs += [
        parent / p
        for p in (
            "manifest.json",
            "observations.npz",
            "selections.json",
            "units.json",
            "head.npz",
            "verification_seal.json",
        )
    ]
    inputs += [parent / "batches" / f"{3 * u:03d}.npz" for u in range(24)]
    inputs += [
        law / p for p in ("manifest.json", "cold_scores.json", "verification_seal.json")
    ]
    inputs += [
        comparison / p
        for p in ("manifest.json", "summary.json", "verification_seal.json")
    ]
    (out / "batches").mkdir(parents=True, exist_ok=False)
    shutil.copyfile(parent / "head.npz", out / "head.npz")
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "alphabet": alphabet,
            "model_files": model_files,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "application_labels_read": False,
            "original_scope_complete": False,
            "backend": "fixed full-key row-chunk256;not uniform hidden-state equivalence",
        },
    )
    R.write(out / "units.json", units)
    R.write(out / "bindings.json", bindings)
    R.write(out / "event_occurrences.json", occurrences)
    R.write(
        out / "source_law_reference.json",
        {
            "path": cfg["source_law"],
            "reference_sha256": R.digest(law / "cold_scores.json"),
            "uses_own_target_vectors_only": True,
            "no_background_reweighting_or_fit": True,
            "unchanged_by_contract_not_empirical_discovery": True,
        },
    )
    R.archive(
        out / "observations.npz",
        **obs,
        source_means=source_means,
        partners=M.partners(),
    )
    R.table(out / "prompt_lengths.csv", quality)
    R.write(
        out / "input_quality.json",
        {
            "authors": 24,
            "unique_target_events": 288,
            "quote_occurrence_blocks": 72,
            "enquiries": 360,
            "new_batches": 72,
            "prompt_tokens_range": [
                min(r["tokens"] for r in quality),
                max(r["tokens"] for r in quality),
            ],
            "full_histories_no_truncation": True,
            "all120_alone_token_prompts_exact": True,
            "target_block_unchanged": True,
            "original_system_alphabet_question_retained": True,
            "application_labels_read": False,
            "new_embeddings_training_API": 0,
            "cyclic_partner_reuse_not_independent_endpoints": True,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        (
            "head.npz",
            "units.json",
            "bindings.json",
            "event_occurrences.json",
            "source_law_reference.json",
            "observations.npz",
            "prompt_lengths.csv",
            "input_quality.json",
        ),
    )
    H.status(out, {"status": "PREPARED", "completed_batches": 0})
    print(
        "HM230 PREPARED:24authors/288events/72batches/360endpoints;original task retained",
        flush=True,
    )


def collect(out):
    """Read each frozen endpoint once; preserve partial failures without restart."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    manifest = R.check(out)
    cfg = manifest["config"]
    if R.read(out / "runtime.json")["status"] != "PREPARED":
        raise ValueError("No automatic retry of an uncertain or terminal job")
    for name, digest in manifest["model_files"].items():
        if R.digest(MODEL / name) != digest:
            raise ValueError("Changed fixed model")
    lock = out / "ACTIVE.lock"
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    elapsed, completed, pending = 0.0, 0, None
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    try:
        torch.set_num_threads(2)
        stamp = time.monotonic()
        observer = DecoderObserver.load(
            str(MODEL),
            device=cfg["device"],
            dtype=cfg["dtype"],
            max_tokens=cfg["max_tokens"],
        )
        F.install(observer.model, cfg["query_rows"])
        load_seconds = time.monotonic() - stamp
        alpha = manifest["alphabet"]["token_ids"]
        if not np.array_equal(
            observer.model.lm_head.weight.detach()[alpha].float().cpu().numpy(),
            H.selected(out / "head.npz", ("weights",))["weights"],
        ):
            raise ValueError("Changed digit readout head")
        units = R.read(out / "units.json")
        for bi in range(cfg["batches"]):
            if elapsed >= cfg["max_model_seconds"]:
                raise RuntimeError("Finite model cap reached; preserve, no extension")
            pending = bi
            H.status(
                out,
                {
                    "status": "RUNNING",
                    "pid": os.getpid(),
                    "completed_batches": completed,
                    "pending_batch": pending,
                    "model_seconds": elapsed,
                },
            )
            batch = units[bi * 5 : (bi + 1) * 5]
            if (
                len(batch) != 5
                or [b["target"] for b in batch] != list(range(5))
                or len({(b["author"], b["mode"]) for b in batch}) != 1
            ):
                raise ValueError("Changed author-mode endpoint binding")
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
                    raise ValueError("Nonfinite reading; no endpoint filtering")
                R.archive(
                    out / "batches" / f"{bi:03d}.npz",
                    logits=logits[:, alpha],
                    normalizer=logsumexp(logits, -1),
                    states=states,
                    greedy=logits.argmax(-1),
                )
            finally:
                elapsed += time.monotonic() - stamp
            completed += 1
            pending = None
            H.status(
                out,
                {
                    "status": "RUNNING",
                    "pid": os.getpid(),
                    "completed_batches": completed,
                    "pending_batch": None,
                    "model_seconds": elapsed,
                },
            )
            print(f"HM230 {completed}/72;model_seconds={elapsed:.3f}", flush=True)
            del body, state, logits, states, ids, mask
            torch.mps.empty_cache()
        H.status(
            out,
            {
                "status": "COLLECTED",
                "pid": None,
                "completed_batches": completed,
                "model_seconds": elapsed,
                "load_seconds": load_seconds,
                "new_endpoints": 360,
                "new_API_private_training_generation_calls": 0,
            },
        )
        R.seal(
            out,
            "collected_seal.json",
            ("runtime.json",) + tuple(f"batches/{i:03d}.npz" for i in range(72)),
        )
    except BaseException as exc:
        H.status(
            out,
            {
                "status": "FAILED",
                "pid": None,
                "completed_batches": completed,
                "pending_batch": pending,
                "model_seconds": elapsed,
                "error": repr(exc),
            },
        )
        raise
    finally:
        lock.unlink(missing_ok=True)


def captured(out):
    """Load all registered complete batches, never impute a missing endpoint."""
    rows = [
        H.selected(
            out / "batches" / f"{bi:03d}.npz", ("logits", "normalizer", "greedy")
        )
        for bi in range(72)
    ]
    return {
        k: np.stack([r[k] for r in rows]).reshape((24, 3, 5) + rows[0][k].shape[1:])
        for k in rows[0]
    }


def analyze(out):
    """Retain all target relations, signed increments and conditional row bands."""
    manifest = R.check(out, STAGES[:1])
    cfg = manifest["config"]
    z = captured(out)
    read = M.readouts(**z, alphabet=manifest["alphabet"]["token_ids"])
    rows, contacts, covariance, summaries = [], [], [], {}
    arrays = dict(z) | read
    old = [
        H.selected(ROOT / cfg["parent"] / "batches" / f"{3 * u:03d}.npz", ("logits",))
        for u in range(24)
    ]
    old_z = np.stack([r["logits"] for r in old])
    old_odds = logsumexp(old_z[..., 5:], -1) - logsumexp(old_z[..., :5], -1)
    for name in ("log_odds", "score"):
        x = read[name]
        a = M.accounts(x)
        arrays[name + "_pieces"] = a["pieces"]
        arrays[name + "_cross_covariance"] = a["cross_covariance"]
        arrays[name + "_other_covariance"] = a["other_covariance"]
        descriptors = []
        for mode, label in enumerate(M.MODES):
            g = M.geometry(x[:, mode])
            arrays[name + "_" + label + "_cosine"] = g["cosine"]
            for q in range(5):
                for r in range(q + 1, 5):
                    contacts.append(
                        {
                            "readout": name,
                            "mode": label,
                            "left": cfg["targets"][q],
                            "right": cfg["targets"][r],
                            "centered_contact": finite(g["cosine"][q, r]),
                        }
                    )
            descriptors.append(
                {
                    "mode": label,
                    "EN_contact": finite(g["cosine"][0, 1]),
                    "first_squared_singular_ratio": g["first_squared_singular_ratio"],
                    "effective_rank": g["effective_rank"],
                    "rms": g["rms"].tolist(),
                }
            )
        for i, left in enumerate(
            ("alone", "duplicate_increment", "other_minus_duplicate")
        ):
            for j, right in enumerate(
                ("alone", "duplicate_increment", "other_minus_duplicate")
            ):
                for q in range(5):
                    for r in range(5):
                        covariance.append(
                            {
                                "readout": name,
                                "left_component": left,
                                "right_component": right,
                                "left_target": cfg["targets"][q],
                                "right_target": cfg["targets"][r],
                                "covariance": float(a["cross_covariance"][i, j, q, r]),
                            }
                        )
        boot = M.bootstrap_en(x, cfg["bootstrap_draws"], cfg["bootstrap_seed"])
        arrays[name + "_EN_row_resamples"] = boot
        en = [d["EN_contact"] for d in descriptors]
        delta = (
            [en[1] - en[0], en[2] - en[1]]
            if all(v is not None for v in en)
            else [None, None]
        )
        summaries[name] = {
            "geometry": descriptors,
            "EN_delta_duplicate_minus_alone": delta[0],
            "EN_delta_other_minus_duplicate": delta[1],
            "EN_conditional_row_resampling_band": [
                [finite(v) for v in np.nanquantile(boot[:, j], [0.025, 0.975])]
                for j in (3, 4)
            ],
            "mean_increments": a["pieces"][:, 1:].mean(axis=0).tolist(),
            "increment_RMS": np.sqrt(np.mean(a["pieces"][:, 1:] ** 2, axis=0)).tolist(),
            "signed_closure_max": a["closure_max"],
            "covariance_closure_max": float(
                np.max(abs(a["cross_covariance"].sum((0, 1)) - a["other_covariance"]))
            ),
        }
    for u in range(24):
        for m, label in enumerate(M.MODES):
            for q, target in enumerate(cfg["targets"]):
                rows.append(
                    {
                        "author": u,
                        "mode": label,
                        "target": target,
                        "log_odds": float(read["log_odds"][u, m, q]),
                        "score": float(read["score"][u, m, q]),
                        "candidate_mass": float(read["candidate_mass"][u, m, q]),
                        "greedy_compliant": bool(read["greedy_compliant"][u, m, q]),
                    }
                )
    summaries["old_alone_readback"] = {
        "log_odds_RMS": float(
            np.sqrt(np.mean((old_odds - read["log_odds"][:, 0]) ** 2))
        ),
        "log_odds_max": float(np.max(abs(old_odds - read["log_odds"][:, 0]))),
        "old_EN_contact": finite(M.geometry(old_odds)["cosine"][0, 1]),
        "not_optimized_equivalence_gate": True,
    }
    summary = {
        "packet": "HM230",
        "observations": {
            "authors": 24,
            "unique_target_events": 288,
            "batches": 72,
            "enquiries": 360,
        },
        "results": summaries,
        "greedy_compliance_by_mode": read["greedy_compliant"]
        .mean(axis=(0, 2))
        .tolist(),
        "candidate_mass_range_by_mode": [
            [
                float(read["candidate_mass"][:, m].min()),
                float(read["candidate_mass"][:, m].max()),
            ]
            for m in range(3)
        ],
        "source_author_law_unchanged_by_contract": True,
        "row_resampling_not_population_CI_due_cyclic_background_reuse": True,
        "not_personality_accuracy_or_unique_semantic_cause": True,
        "no_application_labels_fit_sign_flip_or_parameter_search": True,
        "runtime": R.read(out / "runtime.json"),
        "original_scope_complete": False,
    }
    R.archive(out / "analysis.npz", **arrays)
    R.table(out / "endpoint_readings.csv", rows)
    R.table(out / "target_contacts.csv", contacts)
    R.table(out / "signed_covariance_accounts.csv", covariance)
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "analysis.npz",
            "endpoint_readings.csv",
            "target_contacts.csv",
            "signed_covariance_accounts.csv",
            "summary.json",
        ),
    )
    print(
        "HM230 ANALYZED:all3modes/all5targets/all signed cross terms;no accuracy selection",
        flush=True,
    )


def verify(out):
    """Independently recompute digit laws, head readings and full covariance."""
    manifest = R.check(out, STAGES[:2])
    cfg = manifest["config"]
    obs = R.prepared(out)
    a = H.selected(
        out / "analysis.npz",
        (
            "log_odds",
            "score",
            "probabilities",
            "candidate_mass",
            "log_odds_pieces",
            "log_odds_cross_covariance",
            "score_pieces",
            "score_cross_covariance",
        ),
    )
    head = H.selected(out / "head.npz", ("weights",))["weights"].astype(float)
    errors = {
        "head": 0.0,
        "digit_law": 0.0,
        "score": 0.0,
        "mass": 0.0,
        "log_odds": 0.0,
        "signed_closure": 0.0,
        "covariance": 0.0,
        "contact": 0.0,
    }
    for bi in range(72):
        batch = H.selected(
            out / "batches" / f"{bi:03d}.npz", ("logits", "normalizer", "states")
        )
        errors["head"] = max(
            errors["head"],
            float(
                np.max(abs(batch["states"].astype(float) @ head.T - batch["logits"]))
            ),
        )
        u, m = divmod(bi, 3)
        for q in range(5):
            z = batch["logits"][q].astype(float)
            c = float(z.max())
            exp_z = np.array([np.exp(float(v) - c) for v in z])
            p = exp_z / sum(exp_z)
            odds = float(np.log(sum(exp_z[5:])) - np.log(sum(exp_z[:5])))
            score = float(sum(float(v) * j / 9 for j, v in enumerate(p)))
            mass = float(np.exp(c + np.log(sum(exp_z)) - batch["normalizer"][q]))
            errors["digit_law"] = max(
                errors["digit_law"], float(np.max(abs(p - a["probabilities"][u, m, q])))
            )
            for name, value in (("log_odds", odds), ("score", score), ("mass", mass)):
                key = "candidate_mass" if name == "mass" else name
                errors[name] = max(errors[name], abs(value - a[key][u, m, q]))
    for name in ("log_odds", "score"):
        x = a[name]
        pieces = a[name + "_pieces"]
        errors["signed_closure"] = max(
            errors["signed_closure"], float(np.max(abs(pieces.sum(1) - x[:, 2])))
        )
        for i in range(3):
            for j in range(3):
                for q in range(5):
                    for r in range(5):
                        left, right = pieces[:, i, q], pieces[:, j, r]
                        value = (
                            sum(
                                float(v - left.mean()) * float(w - right.mean())
                                for v, w in zip(left, right, strict=True)
                            )
                            / 24
                        )
                        errors["covariance"] = max(
                            errors["covariance"],
                            abs(value - a[name + "_cross_covariance"][i, j, q, r]),
                        )
        for m, label in enumerate(M.MODES):
            saved = H.selected(out / "analysis.npz", (name + "_" + label + "_cosine",))[
                name + "_" + label + "_cosine"
            ]
            alternate = np.corrcoef(x[:, m], rowvar=False)
            errors["contact"] = max(
                errors["contact"], float(np.nanmax(abs(saved - alternate)))
            )
    units = R.read(out / "units.json")
    old_units = R.read(ROOT / cfg["parent"] / "units.json")
    original = all(
        units[u * 15 + q]["input_ids"] == old_units[u * 15 + q]["input_ids"]
        for u in range(24)
        for q in range(5)
    )
    occurrences = R.read(out / "event_occurrences.json")
    identity = all(
        r["target_event_ids"] == occurrences[r["author"] * 3]["target_event_ids"]
        and len(set(r["target_event_ids"])) == 12
        for r in occurrences
    )
    duplicate = all(
        r["background_event_ids"] == r["target_event_ids"]
        for r in occurrences
        if r["mode"] == "duplicate"
    )
    source_records = R.read(ROOT / cfg["source_law"] / "cold_scores.json")
    source_error = float(
        np.max(
            abs(obs["source_means"] - np.array([r["mean"][0] for r in source_records]))
        )
    )
    result = {
        "PASS": errors["head"] <= cfg["head_tolerance"]
        and all(v <= 1e-10 for k, v in errors.items() if k != "head")
        and original
        and identity
        and duplicate
        and source_error == 0,
        "max_errors": errors,
        "original120_alone_prompts": original,
        "target_event_identity": identity,
        "duplicates_not_new_events": duplicate,
        "fixed_source_reference_error": source_error,
        "endpoints_checked": 360,
        "cross_covariance_entries_checked": 450,
        "same_agent_not_blind": True,
        "no_new_forwards_or_refit": True,
    }
    R.write(out / "verification.json", result)
    R.seal(out, "verification_seal.json", ("verification.json",))
    if not result["PASS"]:
        raise ValueError("Verification failure preserved; no scientific retry")
    print(
        "HM230 VERIFIED:360digit laws/450cross accounts/original task and event identities",
        flush=True,
    )


def main():
    """Run only an explicit stage; no implicit experiment queue or retries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "collect", "analyze", "verify", "check")
    )
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        R.check(args.out, tuple(s for s in STAGES if (args.out / s).exists()))
        print("HM230 existing seals PASS;no experiment started")
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
