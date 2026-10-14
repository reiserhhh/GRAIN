#!/usr/bin/env python3
"""Observe aligned two-window evidence once, without predictor or prompt search."""

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
from grain_research.construction import math_companion_observation as M
from grain_research.construction import math_relational_target_operator as T
from grain_research.construction import math_target_relation_operator as C
from grain_research.vector_mechanisms import direct_traits as D
from grain_research.vector_mechanisms.navigation import MODEL
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_relational_target_operator_hm207 as P
from scripts import run_grain_same_event_multitarget_hm199 as N
from scripts import run_grain_target_relation_operator_hm242 as Q

CONFIG = "configs/grain_companion_observation_hm243.json"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_COMPANION_OBSERVATION_PLAN_HM243.md",
    "grain_research/construction/math_companion_observation.py",
    "scripts/run_grain_companion_observation_hm243.py",
    "tests/test_grain_companion_observation.py",
    "grain_research/vector_mechanisms/direct_traits.py",
    "grain_research/vector_mechanisms/trace.py",
    "grain_research/vector_mechanisms/navigation.py",
    "grain_research/construction/math_relational_target_operator.py",
    "grain_research/construction/math_target_relation_operator.py",
    "scripts/run_grain_observed_family_hm170.py",
    "scripts/run_grain_joint_target_binding_hm198.py",
)
DEFAULT = ROOT / "results/grain_math_response_v1/companion_observation_r1"


def arrays(path, keys=None):
    """Read an explicit numeric support without pickle or application scales."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in (z.files if keys is None else keys)}


def mixed_targets(values):
    """Keep target last while contrasting preceding companion and arm axes."""
    z = np.asarray(values, float)
    return z[..., 0, 0, :] - z[..., 0, 1, :] - z[..., 1, 0, :] + z[..., 1, 1, :]


def prepare(out):
    """Freeze old operators, same event addresses and new token observations."""
    from transformers import AutoTokenizer

    cfg = R.read(ROOT / CONFIG)
    rp, cp, np_, mp = (
        ROOT / cfg[k]
        for k in (
            "relation_parent",
            "covariance_parent",
            "native_parent",
            "material_parent",
        )
    )
    R.check(rp, ("built_seal.json", "analyzed_seal.json", "verified_seal.json"))
    R.check(cp, ("computed_seal.json", "verification_seal.json"))
    N.check(np_)
    N.H.check_hashes(np_, R.read(np_ / "analyzed_seal.json"))
    N.H.check_hashes(np_, R.read(np_ / "verification_seal.json"))
    mm = R.read(mp / "manifest.json")
    material = mp / "private_material.json"
    if R.digest(material) != mm["material_sha256"] or mm["arm"] != "cue_mask":
        raise ValueError("Historical public cue-mask material changed")
    z = arrays(
        rp / "observations.npz", ("panel_vectors", "selected", "control", "author_ids")
    )
    nz = arrays(
        np_ / "00/observations.npz",
        ("panel_vectors", "selected", "control", "author_ids"),
    )
    if any(not np.array_equal(z[k], nz[k]) for k in z):
        raise ValueError("Native/source evidence does not have the same event support")
    if z["panel_vectors"].shape != (24, 2, 12, 1024):
        raise ValueError("No silent cohort or event narrowing")
    records = R.read(material)
    lookup = {
        (r["author_id"], r["view"]): r
        for r in records
        if r["source_group"] == "big5_only"
    }
    tok = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    alphabet = D.candidates("extraversion", tok)
    permutation = np.roll(np.arange(24), -1)
    points = np.empty((24, 2, 2, 2, 2, 1024))
    units, events, lengths = [], [], []
    for u, v, companion, arm in np.ndindex(24, 2, 2, 2):
        uid, peer = (
            str(z["author_ids"][u]),
            u if companion == 0 else int(permutation[u]),
        )
        focal, other = lookup[(uid, v)], lookup[(str(z["author_ids"][peer]), 1 - v)]
        if any(
            not np.array_equal(focal[k], z[k][u, v]) for k in ("selected", "control")
        ):
            raise ValueError("Changed selected/control event identity")
        removed = z[("selected", "control")[arm]][u, v]
        slots = [i for i in range(12) if i not in removed]
        texts = [focal["texts"][i] for i in slots] + other["texts"]
        left, right = (
            z["panel_vectors"][u, v, slots].mean(0),
            z["panel_vectors"][peer, 1 - v].mean(0),
        )
        points[u, v, companion, arm] = [left, right] if v == 0 else [right, left]
        addresses = [f"{uid}:{v}:{i}" for i in slots] + [
            f"{z['author_ids'][peer]}:{1 - v}:{i}" for i in range(12)
        ]
        if len(set(addresses)) != 20:
            raise ValueError("Repeated evidence within one native inquiry")
        events.append(
            {
                "address": [u, v, companion, arm],
                "event_ids": addresses,
                "actual_companion_author": peer,
                "stranger_is_audit_reassignment": companion == 1,
            }
        )
        prefix = None
        for q, task in enumerate(cfg["targets"]):
            ids, user, _ = D.tokenize_prompt(
                texts, task, tok, max_tokens=cfg["max_tokens"]
            )
            rendered = tok.apply_chat_template(
                [
                    {"role": "system", "content": D.SYSTEM},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = tok(
                rendered, add_special_tokens=False, return_offsets_mapping=True
            )
            boundary = rendered.rindex("\nEvidence ends here.")
            pos = max(
                i
                for i, (a, b) in enumerate(encoded["offset_mapping"])
                if boundary >= b > a
            )
            if encoded["input_ids"] != ids or (
                prefix is not None and prefix != ids[: pos + 1]
            ):
                raise ValueError("Target query changed the causal evidence prefix")
            prefix = ids[: pos + 1]
            units.append(
                {
                    "author": u,
                    "view": v,
                    "companion": companion,
                    "arm": arm,
                    "target": q,
                    "input_ids": ids,
                    "body_position": pos,
                }
            )
            lengths.append(len(ids))
    if len(units) != cfg["new_enquiries"]:
        raise ValueError("Incomplete declared observation")
    bs, basis = P.banks(rp)
    flat = points.reshape(-1, 2, 1024)
    source = np.array(
        [
            [
                T.read(bank, component, "full", flat[:, 0], flat[:, 1], basis)[
                    :, T.TARGET_MODES
                ]
                for component in T.COMPONENTS
            ]
            for bank in bs
        ]
    ).reshape(2, 3, 24, 2, 2, 2, 5)
    cov = (
        np.array([C.score_points(bank, flat) for bank in Q.load_banks(cp)])
        .reshape(2, 24, 2, 2, 2, 3, 6)
        .transpose(0, 5, 1, 2, 3, 4, 6)
    )
    old = arrays(np_ / "analysis.npz", ("native_actions",))["native_actions"][
        :, :, 0, :, :2
    ]
    files = [
        rp / "observations.npz",
        rp / "operator_bundle.npz",
        rp / "manifest.json",
        cp / "source_banks.npz",
        cp / "manifest.json",
        np_ / "00/observations.npz",
        np_ / "analysis.npz",
        np_ / "00/manifest.json",
        material,
        mp / "manifest.json",
    ]
    model_files = {
        p.name: R.digest(p)
        for p in MODEL.iterdir()
        if p.is_file() and not p.name.startswith("._")
    }
    if model_files["model.safetensors"] != mm["model_sha256"]:
        raise ValueError("Different frozen native observer")
    out.mkdir(parents=True, exist_ok=False)
    (out / "01").mkdir()
    R.archive(
        out / "observations.npz",
        points=points,
        source_corners=source,
        covariance_corners=cov,
        old_single_actions=old,
        permutation=permutation,
        **z,
    )
    R.write(out / "units.json", units)
    R.write(out / "event_contracts.json", events)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in files},
            "model_files": model_files,
            "alphabet": alphabet,
            "public_material_only": True,
            "application_labels_read": False,
            "new_human_authors": 0,
            "original_scope_complete": False,
        },
    )
    quality = {
        "authors": 24,
        "unique_source_events": 576,
        "endpoint_inquiries": len(units),
        "native_excerpts_per_inquiry": 20,
        "tokens_min": min(lengths),
        "tokens_median": float(np.median(lengths)),
        "tokens_max": max(lengths),
        "new_prompt_grid": False,
        "source_actions_frozen_before_native": True,
        "stranger_is_not_a_real_author_history": True,
        "old_protocol_observed_companion": False,
    }
    R.write(out / "quality.json", quality)
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "units.json", "event_contracts.json", "quality.json"),
    )
    N.H.status_write(
        out,
        {
            "state": "PREPARED",
            "completed": 0,
            "attempts": 0,
            "pending": None,
            "model_seconds": 0.0,
        },
    )
    print(json.dumps(quality), flush=True)


def collect(out):
    """Collect each frozen actual two-window observation once within the cap."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    m = R.check(out)
    cfg, status = m["config"], R.read(out / "status.json")
    if status["state"] != "PREPARED" or status["pending"] is not None:
        raise RuntimeError("No uncertain/failed/terminal automatic restart")
    N.H.check_hashes(MODEL, m["model_files"])
    units = R.read(out / "units.json")
    lock = out / "collector.lock"
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    try:
        torch.set_num_threads(2)
        start = time.monotonic()
        observer = DecoderObserver.load(
            str(MODEL),
            device=cfg["device"],
            dtype=cfg["dtype"],
            max_tokens=cfg["max_tokens"],
        )
        status["model_seconds"] += time.monotonic() - start
        N.H.status_write(out, status)
        R.write(
            out / "runtime.json",
            {
                "pid": os.getpid(),
                "torch": torch.__version__,
                "device": cfg["device"],
                "dtype": cfg["dtype"],
            },
        )
        device, alphabet = (
            next(observer.model.parameters()).device,
            m["alphabet"]["token_ids"],
        )
        head = observer.model.lm_head
        R.archive(
            out / "01/head.npz",
            weights=head.weight.detach()[alphabet].float().cpu().numpy(),
            bias=np.zeros(10)
            if head.bias is None
            else head.bias.detach()[alphabet].float().cpu().numpy(),
        )
        for b in range(cfg["new_batches"]):
            if status["model_seconds"] >= cfg["max_model_seconds"]:
                status["state"] = "BUDGET_STOP"
                raise TimeoutError("No automatic finite cap renewal")
            batch = units[5 * b : 5 * b + 5]
            address = {
                (u["author"], u["view"], u["companion"], u["arm"]) for u in batch
            }
            if len(address) != 1 or [u["target"] for u in batch] != list(range(5)):
                raise ValueError("Batch query/address contract changed")
            status.update(state="RUNNING", attempts=status["attempts"] + 1, pending=b)
            N.H.status_write(out, status)
            start = time.monotonic()
            try:
                length = max(len(u["input_ids"]) for u in batch)
                ids = torch.tensor(
                    [
                        u["input_ids"]
                        + [observer.tokenizer.pad_token_id]
                        * (length - len(u["input_ids"]))
                        for u in batch
                    ],
                    device=device,
                )
                mask = torch.tensor(
                    [
                        [1] * len(u["input_ids"]) + [0] * (length - len(u["input_ids"]))
                        for u in batch
                    ],
                    device=device,
                )
                ix = torch.arange(5, device=device)
                last = torch.tensor(
                    [len(u["input_ids"]) - 1 for u in batch], device=device
                )
                body = torch.tensor([u["body_position"] for u in batch], device=device)
                captured = {}

                def capture(
                    _module, _args, output, captured=captured, ix=ix, body=body
                ):
                    h = output[0] if isinstance(output, tuple) else output
                    captured["body"] = h[ix, body].detach().float().cpu().numpy()

                hook = observer.model.model.layers[
                    cfg["capture_layer"]
                ].register_forward_hook(capture)
                try:
                    with torch.inference_mode():
                        result = observer.model.model(
                            input_ids=ids,
                            attention_mask=mask,
                            use_cache=False,
                            return_dict=True,
                        )
                        h = result.last_hidden_state[ix, last]
                        full = (
                            observer.model.lm_head(h)
                            .float()
                            .cpu()
                            .numpy()
                            .astype(float)
                        )
                        h = h.detach().float().cpu().numpy()
                finally:
                    hook.remove()
                err = float(np.max(abs(captured["body"] - captured["body"][0])))
                if err > cfg["head_tolerance"]:
                    raise ValueError("Future question changed common causal evidence")
                path = out / f"01/b{b:03d}.npz"
                R.archive(
                    path,
                    logits=full[:, alphabet],
                    normalizer=logsumexp(full, axis=1),
                    states=h,
                )
                R.write(
                    path.with_suffix(".json"),
                    {
                        "address": list(next(iter(address))),
                        "sha256": R.digest(path),
                        "prefix_error": err,
                    },
                )
            finally:
                status["model_seconds"] += time.monotonic() - start
                N.H.status_write(out, status)
            status.update(completed=b + 1, pending=None)
            N.H.status_write(out, status)
            if (b + 1) % 12 == 0:
                print(
                    f"HM243 {b + 1}/{cfg['new_batches']} batches;{status['model_seconds']:.3f}s",
                    flush=True,
                )
        R.seal(
            out,
            "collected_seal.json",
            [p.relative_to(out).as_posix() for p in sorted((out / "01").iterdir())]
            + ["runtime.json"],
        )
        status.update(state="COLLECTED", pending=None)
        N.H.status_write(out, status)
    except Exception as exc:
        if status["state"] != "BUDGET_STOP":
            status["state"] = "FAILED"
        status["error"] = repr(exc)
        N.H.status_write(out, status)
        raise
    finally:
        lock.unlink()


def collected(out):
    """Read actual, complete endpoint arrays with recorded identities."""
    m = R.check(out, ("collected_seal.json",))
    status = R.read(out / "status.json")
    if (
        status["state"] != "COLLECTED"
        or status["completed"] != m["config"]["new_batches"]
    ):
        raise RuntimeError("Incomplete native support")
    shape = (24, 2, 2, 2, 5)
    logits, normalizer, states = (
        np.empty((*shape, 10)),
        np.empty(shape),
        np.empty((*shape, 1536)),
    )
    units = R.read(out / "units.json")
    for b in range(192):
        a = arrays(out / f"01/b{b:03d}.npz")
        for j, unit in enumerate(units[5 * b : 5 * b + 5]):
            key = tuple(
                unit[k] for k in ("author", "view", "companion", "arm", "target")
            )
            logits[key], normalizer[key], states[key] = (
                a["logits"][j],
                a["normalizer"][j],
                a["states"][j],
            )
    return m, logits, normalizer, states


def analyze(out):
    """Report all interactions/contacts without selecting a favorable target."""
    m, logits, normalizer, states = collected(out)
    cfg, z, head = m["config"], R.prepared(out), arrays(out / "01/head.npz")
    readings = M.read_logits(logits)
    native = {k: mixed_targets(readings[k]) for k in ("score", "log_odds")}
    h = states.transpose(0, 1, 4, 2, 3, 5)
    accounts = M.head_accounts(h, head["weights"], head["bias"])
    source, covariance = (
        mixed_targets(z["source_corners"]),
        mixed_targets(z["covariance_corners"]),
    )
    head_error = float(
        np.max(abs(accounts["logits"] - logits.transpose(0, 1, 4, 2, 3, 5)))
    )
    rows, endpoint_rows, contact_rows = [], [], []
    masses = np.exp(readings["log_candidate_partition"] - normalizer)
    for u, v, q in np.ndindex(24, 2, 5):
        action = (
            readings["log_odds"][u, v, :, 0, q] - readings["log_odds"][u, v, :, 1, q]
        )
        rows.append(
            {
                "author": u,
                "view": v,
                "target": cfg["targets"][q],
                "own_action": float(action[0]),
                "stranger_action": float(action[1]),
                "old_alone_action": float(z["old_single_actions"][u, v, q, 1]),
                "mixed_log_odds": float(native["log_odds"][u, v, q]),
                "mixed_score": float(native["score"][u, v, q]),
                "head_reader": float(accounts["reader"][u, v, q]),
                "head_representation": float(accounts["representation"][u, v, q]),
                "native_minus_head": float(
                    native["log_odds"][u, v, q] - accounts["total"][u, v, q]
                ),
                "mixed_centered_logit_rms": M.rms(
                    accounts["mixed_centered_logits"][u, v, q]
                ),
            }
        )
    for u, v, c, a, q in np.ndindex(24, 2, 2, 2, 5):
        endpoint_rows.append(
            {
                "author": u,
                "view": v,
                "companion": cfg["companions"][c],
                "arm": cfg["arms"][a],
                "target": cfg["targets"][q],
                "log_odds": float(readings["log_odds"][u, v, c, a, q]),
                "score": float(readings["score"][u, v, c, a, q]),
                "candidate_mass": float(masses[u, v, c, a, q]),
            }
        )
    for family, values, components, targets in (
        ("uniform32_relation", source, T.COMPONENTS, cfg["targets"]),
        ("empirical_covariance", covariance, C.COMPONENTS, C.TARGETS),
    ):
        for bank, component in np.ndindex(2, 3):
            for unit, y in native.items():
                table = M.contacts(values[bank, component], y)
                for t in range(len(targets)):
                    for q in range(5):
                        contact_rows.append(
                            {
                                "family": family,
                                "bank": bank,
                                "component": components[component],
                                "unit": unit,
                                "source_target": targets[t],
                                "query_target": cfg["targets"][q],
                                "centered_contact": table[t][q],
                                "mean_signed_product": float(
                                    np.mean(values[bank, component, ..., t] * y[..., q])
                                ),
                            }
                        )
    rng, intervals = np.random.default_rng(cfg["bootstrap_seed"]), []
    draws = rng.integers(24, size=(cfg["bootstrap_draws"], 24))
    for bank, target in np.ndindex(2, 5):
        samples = [
            M.cosine(source[bank, 2, ix, :, target], native["log_odds"][ix, :, target])
            for ix in draws
        ]
        samples = np.array([x for x in samples if x is not None])
        intervals.append(
            {
                "bank": bank,
                "target": cfg["targets"][target],
                "contact": M.cosine(
                    source[bank, 2, ..., target], native["log_odds"][..., target]
                ),
                "conditional_author_bootstrap": np.quantile(
                    samples, [0.025, 0.975]
                ).tolist(),
                "not_independent_population_CI": True,
            }
        )
    summaries = []
    for q, target in enumerate(cfg["targets"]):
        y = readings["log_odds"][..., q]
        own, stranger = y[:, :, 0, 0] - y[:, :, 0, 1], y[:, :, 1, 0] - y[:, :, 1, 1]
        old = z["old_single_actions"][..., q, 1]
        summaries.append(
            {
                "target": target,
                "mixed_rms": M.rms(native["log_odds"][..., q]),
                "reader_rms": M.rms(accounts["reader"][..., q]),
                "representation_rms": M.rms(accounts["representation"][..., q]),
                "signed_reader_representation_product": float(
                    np.mean(
                        accounts["reader"][..., q] * accounts["representation"][..., q]
                    )
                ),
                "own_action_rms": M.rms(own),
                "stranger_action_rms": M.rms(stranger),
                "alone_action_rms": M.rms(old),
                "own_vs_alone_contact": M.cosine(own, old),
                "stranger_vs_alone_contact": M.cosine(stranger, old),
                "own_vs_stranger_contact": M.cosine(own, stranger),
                "same_action_direction_count": int(np.sum(own * stranger > 0)),
                "observations": 48,
                "authors": 24,
            }
        )
    R.archive(
        out / "accounts.npz",
        native_logits=logits,
        normalizer=normalizer,
        states=states,
        mixed_log_odds=native["log_odds"],
        mixed_score=native["score"],
        source_mixed=source,
        covariance_mixed=covariance,
        **{
            k: accounts[k]
            for k in (
                "reader",
                "representation",
                "total",
                "mixed_hidden",
                "mixed_centered_logits",
            )
        },
    )
    R.table(out / "native_actions.csv", rows)
    R.table(out / "native_endpoints.csv", endpoint_rows)
    R.table(out / "source_contacts.csv", contact_rows)
    R.write(
        out / "summary.json",
        {
            "packet": "HM243",
            "targets": summaries,
            "primary_pair_contacts": intervals,
            "head_error": head_error,
            "head_error_pass": head_error <= cfg["head_tolerance"],
            "head_closure_error": float(np.max(abs(accounts["closure"]))),
            "native_head_mixed_error": float(
                np.max(abs(native["log_odds"] - accounts["total"]))
            ),
            "candidate_mass_range": [float(masses.min()), float(masses.max())],
            "collection": R.read(out / "status.json"),
            "not_prediction_accuracy_or_new_trait": True,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "accounts.npz",
            "native_actions.csv",
            "native_endpoints.csv",
            "source_contacts.csv",
            "summary.json",
        ),
    )
    print(
        json.dumps(
            {
                "targets": summaries,
                "primary_pair_contacts": intervals,
                "head_error": head_error,
            }
        ),
        flush=True,
    )


def verify(out):
    """Independent scalar/head/field accounts, without new model observations."""
    m = R.check(out, ("collected_seal.json", "analyzed_seal.json"))
    z, a, head = (
        R.prepared(out),
        arrays(out / "accounts.npz"),
        arrays(out / "01/head.npz"),
    )
    bs, basis = P.banks(ROOT / m["config"]["relation_parent"])
    logits = a["native_logits"]
    shift = logits.max(axis=-1, keepdims=True)
    mass = np.exp(logits - shift)
    scalar = np.log(mass[..., 5:].sum(-1) / mass[..., :5].sum(-1))
    native_err = float(np.max(abs(mixed_targets(scalar) - a["mixed_log_odds"])))
    field_err = 0.0
    for bank, component, u, v in np.ndindex(2, 3, 24, 2):
        own_s, own_c = z["points"][u, v, 0, 0, v], z["points"][u, v, 0, 1, v]
        own_b, foreign_b = (
            z["points"][u, v, 0, 0, 1 - v],
            z["points"][u, v, 1, 0, 1 - v],
        )
        fields = T.field(
            bs[bank], T.COMPONENTS[component], "full", own_b, v, basis
        ) - T.field(bs[bank], T.COMPONENTS[component], "full", foreign_b, v, basis)
        literal = fields[T.TARGET_MODES] @ (own_s - own_c)
        field_err = max(
            field_err,
            float(np.max(abs(literal - a["source_mixed"][bank, component, u, v]))),
        )
    corner_err = 0.0
    for u, v, q in np.ndindex(24, 2, 5):
        h = a["states"][u, v, :, :, q]
        zz = h @ head["weights"].astype(float).T + head["bias"]
        f = lambda zz: float(
            np.log(np.exp(zz[5:] - zz.max()).sum() / np.exp(zz[:5] - zz.max()).sum())
        )
        ha = h[1, 0] + h[0, 1] - h[1, 1]
        fr = (
            f(ha @ head["weights"].astype(float).T + head["bias"])
            - f(zz[0, 1])
            - f(zz[1, 0])
            + f(zz[1, 1])
        )
        corner_err = max(corner_err, abs(fr - a["reader"][u, v, q]))
    summary = R.read(out / "summary.json")
    receipt = {
        "native_scalar_accounts": 960,
        "relation_field_accounts": 288,
        "head_reader_accounts": 240,
        "native_scalar_error": native_err,
        "relation_field_error": field_err,
        "head_reader_error": corner_err,
        "head_compatibility_error": summary["head_error"],
        "same_agent_alternate_not_blind": True,
        "new_neural_forwards": 0,
    }
    if (
        max(native_err, field_err, corner_err) > m["config"]["arithmetic_tolerance"]
        or not summary["head_error_pass"]
    ):
        R.write(out / "verification_failure.json", receipt)
        raise ValueError("Fixed observation compatibility or arithmetic failure")
    R.write(out / "verification.json", receipt)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(receipt), flush=True)


def main():
    """Explicit stages; no retry, grid, queue or implicit experiment continuation."""
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
        print("HM243 method/input/event/output seals PASS", flush=True)
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
