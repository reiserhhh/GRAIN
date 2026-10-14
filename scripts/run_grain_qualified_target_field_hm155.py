#!/usr/bin/env python3
"""Execute a finite reader-evidence composition experiment, without fitting."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_qualified_target_field as M
from scripts import run_grain_joint_persona_style_hm150 as J
from scripts import run_grain_target_wording_fields_hm154 as OLD

H = J.H
DEFAULT = ROOT / "results/grain_math_response_v1/qualified_reader_fields_r1"
MODES = ("natural", "e_local", "n_local", "both_local", "both_whole")
CODE = (
    "grain_research/construction/math_qualified_target_field.py",
    "scripts/run_grain_qualified_target_field_hm155.py",
    "scripts/verify_grain_qualified_target_field_hm155.py",
    "tests/test_grain_qualified_target_field.py",
    "docs/GRAIN_QUALIFIED_TARGET_FIELD_PLAN_HM155.md",
)
DEPENDENCIES = (
    "scripts/run_grain_joint_persona_style_hm150.py",
    "scripts/run_grain_semantic_anchor_hm147.py",
    "grain_research/vector_mechanisms/trace.py",
    "grain_research/construction/math_native_kv_identity.py",
)
GROUPS = tuple(itertools.product(range(2), repeat=3))


def read(path):
    """Read one strict finite JSON artifact."""
    with Path(path).open() as f:
        return json.load(f)


def arrays(path):
    """Read a local numeric observation without pickle or model execution."""
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k].copy() for k in a.files}


def status(out, state, **values):
    """Write only the mutable owned checkpoint, never overwrite scientific data."""
    J.status(out, "collection", state, **values)


def check(out):
    """Authenticate the prospective design and unchanged code dependencies."""
    manifest = read(out / "manifest.json")
    for name, checksum in manifest["hashes"].items():
        if H.digest(ROOT / name) != checksum:
            raise ValueError("Sealed dependency changed: " + name)
    for name, checksum in manifest["inputs"].items():
        if H.digest(out / name) != checksum:
            raise ValueError("Prepared input changed: " + name)
    return manifest


def prepare(out):
    """Seal all new records, native token bags and interventions before forwards."""
    from transformers import AutoTokenizer

    parent = OLD.check(OLD.DEFAULT)
    old_seal = read(OLD.DEFAULT / "output_seal.json")
    for name, checksum in old_seal.items():
        if H.digest(OLD.DEFAULT / name) != checksum:
            raise ValueError("HM154 output changed: " + name)
    tokenizer = AutoTokenizer.from_pretrained(str(J.MODEL), local_files_only=True)
    prefix = tokenizer.encode(
        "<|im_start|>system\n" + M.SYSTEM + "<|im_end|>\n<|im_start|>user\n",
        add_special_tokens=False,
    )
    rows = M.materials()
    for r in rows:
        encoded = tokenizer(
            r["body"], add_special_tokens=False, return_offsets_mapping=True
        )
        r["input_ids"] = prefix + encoded["input_ids"]
        r["history"] = list(range(len(prefix), len(r["input_ids"])))
        r["scopes"] = {
            axis: [
                j
                for j, (a, b) in enumerate(encoded["offset_mapping"])
                if b > a >= start and b <= stop
            ]
            for axis, (start, stop) in r["intervals"].items()
        }
        if any(not scope for scope in r["scopes"].values()):
            raise ValueError("Empty complete-token evidence scope")
        if set(r["scopes"][M.AXES[0]]) & set(r["scopes"][M.AXES[1]]):
            raise ValueError("Local evidence scopes overlap")
    indexed = {(r["bank"], r["condition"], r["order"], r["e"], r["n"]): r for r in rows}
    for r in rows:
        key = (r["bank"], r["condition"], r["order"])
        for bit, axis in enumerate(M.AXES):
            donor = indexed[
                *key, -r["e"] if bit == 0 else r["e"], -r["n"] if bit == 1 else r["n"]
            ]
            if len(r["input_ids"]) != len(donor["input_ids"]):
                raise ValueError("Actor edit changed native width")
            if Counter(r["input_ids"]) != Counter(donor["input_ids"]):
                raise ValueError("Actor edit changed complete native token bag")
            if r["scopes"] != donor["scopes"]:
                raise ValueError("Actor edit changed clause addresses")
            changed = {
                i - len(prefix)
                for i, (x, y) in enumerate(
                    zip(r["input_ids"], donor["input_ids"], strict=True)
                )
                if x != y
            }
            if not changed or not changed <= set(r["scopes"][axis]):
                raise ValueError("Actor swap escaped its declared clause")
    queries = {
        "definitions": [{"axis": a, "family": f, "person": p} for a, f, p in M.QUERIES],
        "questions": M.questions(),
        "suffixes": [
            tokenizer.encode(
                "\nQuestion: " + q + "\n<|im_end|>\n<|im_start|>assistant\n",
                add_special_tokens=False,
            )
            for q in M.questions()
        ],
        "answers": {
            s: tokenizer.encode(s, add_special_tokens=False) for s in ("Yes", "No")
        },
    }
    if any(len(a) != 1 for a in queries["answers"].values()):
        raise ValueError("This frozen reader requires single native answer tokens")
    maximum = max(
        len(r["input_ids"]) + len(s) for r in rows for s in queries["suffixes"]
    )
    if maximum > 768:
        raise ValueError("No text/query truncation allowed")
    design = [
        {"batch": i, "row": r["id"], "mode": "natural"} for i, r in enumerate(rows)
    ]
    for row_id in (0, 3, 16, 31):
        design.append({"batch": len(design), "row": row_id, "mode": "self"})
    for mode in MODES[1:]:
        for r in rows:
            design.append({"batch": len(design), "row": r["id"], "mode": mode})
    if len(design) != 164:
        raise ValueError("Finite collection support changed")
    out.mkdir(parents=True, exist_ok=False)
    (out / "captures").mkdir()
    (out / "accounts").mkdir()
    for name, value in (("rows", rows), ("queries", queries), ("design", design)):
        H.write_json(out / (name + ".json"), value)
    source = [
        str((OLD.DEFAULT / name).relative_to(ROOT))
        for name in ("manifest.json", "output_seal.json", "alternate_audit.json")
    ]
    model_files = [
        str((J.MODEL / name).relative_to(ROOT))
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json")
    ]
    H.write_json(
        out / "manifest.json",
        {
            "packet": "HM155",
            "hashes": {
                p: H.digest(ROOT / p)
                for p in (*CODE, *DEPENDENCIES, *source, *model_files)
            },
            "inputs": {
                p: H.digest(out / p)
                for p in ("rows.json", "queries.json", "design.json")
            },
            "model_weights_sha256": parent["model_weights_sha256"],
            "maximum_tokens": maximum,
            "collection_seconds_cap": 900,
            "analysis_seconds_cap": 300,
            "audit_seconds_cap": 120,
            "batches": 164,
            "inquiry_sequences": 1968,
            "records": 32,
            "selected_atol": 1e-4,
            "selected_rtol": 1e-5,
            "prefix_atol": 1e-4,
            "fixed_vocab_spots": [[g, f] for g in (0, 7) for f in range(3)],
            "original_scope_complete": False,
            "protocol": "explicit independently tokenized chat/body/suffix fragments",
        },
    )
    status(out, "PREPARED", batches_done=0, pending=None)
    print({"prepared": 32, "batches": 164, "maximum_tokens": maximum}, flush=True)


def native_batch(observer, row, queries, donors=None):
    """Capture final reader states and intervene only on suffix access to K/V."""
    import torch
    import transformers.models.qwen2.modeling_qwen2 as qwen

    from grain_research.construction.math_native_kv_identity import suffix_interchange

    prompts = [row["input_ids"] + s for s in queries["suffixes"]]
    lengths = [len(p) for p in prompts]
    width, batch = max(lengths), len(prompts)
    device = next(observer.model.parameters()).device
    ids = torch.zeros((batch, width), device=device, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, p in enumerate(prompts):
        ids[i, : len(p)] = torch.tensor(p, device=device)
        mask[i, : len(p)] = 1
    body, history = len(row["input_ids"]), row["history"]
    original, captures, calls = qwen.eager_attention_forward, {}, []
    spread = 0.0

    def interface(module, query, key, value, attention_mask, scaling, **kwargs):
        nonlocal spread
        calls.append(module.layer_idx)
        for name, tensor in (("key", key), ("value", value)):
            h = tensor[:, :, history].float().cpu().numpy()
            spread = max(spread, float(np.max(abs(h - h[:1]))))
            if donors is None:
                captures[name, module.layer_idx] = h[0].copy()
        if donors is None:
            return original(
                module, query, key, value, attention_mask, scaling=scaling, **kwargs
            )
        return suffix_interchange(
            original,
            query,
            key,
            value,
            attention_mask,
            scaling,
            body,
            history,
            donors[0][module.layer_idx],
            donors[1][module.layer_idx],
            module,
            **kwargs,
        )

    try:
        qwen.eager_attention_forward = interface
        with torch.inference_mode():
            hidden = observer.model.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=torch.arange(width, device=device)[None].expand(batch, -1),
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
            states = hidden[
                torch.arange(batch, device=device),
                torch.tensor(lengths, device=device) - 1,
            ]
            logits = observer.model.lm_head(states).float()
            answer_ids = [queries["answers"][s][0] for s in ("Yes", "No")]
            selected = logits[:, answer_ids]
            expected = (
                states @ observer.model.get_output_embeddings().weight[answer_ids].T
            )
            if not torch.allclose(selected, expected, atol=1e-4, rtol=1e-5):
                raise ValueError("Native output-head reconstruction changed")
            result = {
                "states": states.float().cpu().numpy().copy(),
                "selected_logits": selected.cpu().numpy().copy(),
                "normalizer": torch.logsumexp(logits, dim=-1).cpu().numpy().copy(),
                "prefix_states": hidden[0, :body].float().cpu().numpy().copy(),
                "cache_spread": np.asarray(spread),
                "blocks": np.asarray(calls),
                "endpoints": np.asarray(lengths) - 1,
            }
    finally:
        qwen.eager_attention_forward = original
    if calls != list(range(len(observer.model.model.layers))):
        raise ValueError("Every native block must execute once")
    if donors is None:
        result["history_key"] = np.stack([captures["key", i] for i in calls])
        result["history_value"] = np.stack([captures["value", i] for i in calls])
    observer.calls += 1
    return result


def donors_for(row, mode, natural, indexed):
    """Construct disjoint local two-donor operations without invented activations."""
    base = natural[row["id"]]
    key, value = base["history_key"].copy(), base["history_value"].copy()
    addr = (row["bank"], row["condition"], row["order"])
    if mode == "both_whole":
        d = natural[indexed[*addr, -row["e"], -row["n"]]["id"]]
        return d["history_key"], d["history_value"]
    for axis, active in zip(
        M.AXES,
        (mode in ("e_local", "both_local"), mode in ("n_local", "both_local")),
        strict=True,
    ):
        if active:
            r = indexed[
                *addr,
                -row["e"] if axis == M.AXES[0] else row["e"],
                -row["n"] if axis == M.AXES[1] else row["n"],
            ]
            d, slots = natural[r["id"]], row["scopes"][axis]
            key[:, :, slots], value[:, :, slots] = (
                d["history_key"][:, :, slots],
                d["history_value"][:, :, slots],
            )
    return key, value


def collect(out):
    """Run all164 finite batches once; save every failure and partial checkpoint."""
    manifest, state = check(out), read(out / "collection_status.json")
    if state["status"] != "PREPARED":
        raise RuntimeError("No automatic replay of partial or terminal experiments")
    start, done, pending = time.monotonic(), 0, None
    rows, queries, design = (
        read(out / (n + ".json")) for n in ("rows", "queries", "design")
    )
    indexed = {(r["bank"], r["condition"], r["order"], r["e"], r["n"]): r for r in rows}
    natural = {}
    try:
        status(out, "LOADING", pid=os.getpid(), batches_done=done, pending=None)
        observer = J.offline_model(manifest)
        H.write_json(out / "model_manifest.json", observer.manifest())
        for entry in design:
            if time.monotonic() - start > manifest["collection_seconds_cap"]:
                raise TimeoutError("Original900second budget exhausted; no renewal")
            pending, row = entry, rows[entry["row"]]
            status(
                out,
                "RUNNING",
                pid=os.getpid(),
                batches_done=done,
                pending=pending,
                seconds=time.monotonic() - start,
            )
            begin = time.monotonic()
            d = (
                None
                if entry["mode"] == "natural"
                else donors_for(row, entry["mode"], natural, indexed)
            )
            result = native_batch(observer, row, queries, d)
            if float(result["cache_spread"]) > manifest["prefix_atol"]:
                raise ValueError("Inquiry changed causal prefix carriers")
            if entry["mode"] == "natural":
                natural[row["id"]] = result
            else:
                base = natural[row["id"]]
                error = float(
                    np.max(abs(result["prefix_states"] - base["prefix_states"]))
                )
                if error > manifest["prefix_atol"]:
                    raise ValueError("Suffix intervention changed prefix")
                result["prefix_error"] = np.asarray(error)
                if entry["mode"] == "self" and not np.allclose(
                    result["selected_logits"],
                    base["selected_logits"],
                    atol=1e-4,
                    rtol=1e-5,
                ):
                    raise ValueError("Own-donor operation changed native observation")
            stem = out / "captures" / f"{entry['batch']:03d}"
            H.write_npz(stem.with_suffix(".npz"), **result)
            H.write_json(
                stem.with_suffix(".json"),
                {
                    **entry,
                    "seconds": time.monotonic() - begin,
                    "sha256": H.digest(stem.with_suffix(".npz")),
                },
            )
            done, pending = done + 1, None
            if done % 8 == 0:
                print(
                    {"HM155_batches": done, "seconds": time.monotonic() - start},
                    flush=True,
                )
        if time.monotonic() - start > manifest["collection_seconds_cap"]:
            raise TimeoutError("Budget crossed on final batch")
        H.write_json(
            out / "capture_seal.json",
            {
                str(p.relative_to(out)): H.digest(p)
                for p in sorted((out / "captures").glob("*"))
            },
        )
        status(
            out,
            "COLLECTED",
            batches_done=done,
            pending=None,
            seconds=time.monotonic() - start,
            model_batches=observer.calls,
            inquiry_sequences=12 * done,
        )
    except BaseException as exc:
        status(
            out,
            "FAILED_OR_BUDGET_STOP",
            batches_done=done,
            pending=pending,
            seconds=time.monotonic() - start,
            error=f"{type(exc).__name__}: {exc}",
            pid=os.getpid(),
        )
        raise


def observation_cubes(out):
    """Read all actual endpoints into complete cubes, never fabricate corners."""
    check(out)
    seal = read(out / "capture_seal.json")
    for name, checksum in seal.items():
        if H.digest(out / name) != checksum:
            raise ValueError("Capture changed: " + name)
    rows, design = read(out / "rows.json"), read(out / "design.json")
    captured = {
        (e["mode"], e["row"]): arrays(out / "captures" / f"{e['batch']:03d}.npz")
        for e in design
    }
    result = {}
    for group in GROUPS:
        for mode in MODES:
            cs = {
                name: np.empty((2, 2, 2, 2, 3, width))
                for name, width in (
                    ("states", 1536),
                    ("selected_logits", 2),
                    ("normalizer", 1),
                )
            }
            for r in rows:
                if (r["bank"], r["condition"], r["order"]) == group:
                    a = captured[mode, r["id"]]
                    for name, cube in cs.items():
                        cube[(r["e"] + 1) // 2, (r["n"] + 1) // 2] = (
                            a[name].reshape(2, 3, 2, -1).transpose(2, 0, 1, 3)
                        )
            cs["logodds"] = (
                cs["selected_logits"][..., 0] - cs["selected_logits"][..., 1]
            )
            cs["answer_mass"] = np.exp(
                logsumexp(cs["selected_logits"], axis=-1) - cs["normalizer"][..., 0]
            )
            result[group, mode] = cs
    return result, captured


def analyze(out):
    """Account qualified responses and local/whole composition, not tune scores."""
    import torch

    start, manifest = time.monotonic(), check(out)
    if read(out / "collection_status.json")["status"] != "COLLECTED":
        raise RuntimeError("Full finite captures required; partials are not completion")
    if (out / "summary.json").exists():
        raise RuntimeError("Do not overwrite primary analysis")
    cubes, captured = observation_cubes(out)
    coefficients, action_rows, composition_rows, field_rows = [], [], [], []
    for group_index, group in enumerate(GROUPS):
        account = {}
        for mode in MODES:
            cs = cubes[group, mode]
            for name in ("states", "selected_logits", "normalizer", "logodds"):
                account[mode + "_" + name] = M.characters(cs[name])
            for axis, family in itertools.product(range(2), range(3)):
                for mask in range(8):
                    coefficients.append(
                        {
                            "group": group_index,
                            "bank": group[0],
                            "condition": group[1],
                            "order": group[2],
                            "mode": mode,
                            "axis": M.AXES[axis],
                            "family": M.FAMILIES[family],
                            "mask": mask,
                            "logodds": account[mode + "_logodds"][mask, axis, family],
                        }
                    )
        for axis in range(2):
            base = cubes[group, "natural"]["logodds"][..., axis, :]
            parts = M.composition(
                base,
                *(cubes[group, mode]["logodds"][..., axis, :] for mode in MODES[1:]),
            )
            for mode in MODES[1:]:
                effect = cubes[group, mode]["logodds"][..., axis, :] - base
                flip = (
                    base[::-1]
                    if mode == "e_local"
                    else base[:, ::-1]
                    if mode == "n_local"
                    else base[::-1, ::-1]
                )
                action_rows.append(
                    dict(
                        group=group_index,
                        bank=group[0],
                        condition=group[1],
                        order=group[2],
                        axis=M.AXES[axis],
                        mode=mode,
                        **M.action_account(effect, flip - base),
                    )
                )
            identity = (
                sum(parts[k] for k in ("e", "n", "interaction", "nonlocal_remainder"))
                - parts["whole"]
            )
            composition_rows.append(
                dict(
                    group=group_index,
                    bank=group[0],
                    condition=group[1],
                    order=group[2],
                    axis=M.AXES[axis],
                    **{
                        k + "_rms": float(np.sqrt(np.mean(v * v)))
                        for k, v in parts.items()
                    },
                    identity_error=float(np.max(abs(identity))),
                )
            )
        natural = cubes[group, "natural"]
        account["reference_state"] = natural["states"].reshape(-1, 1536).mean(axis=0)
        account["qualified_fields"] = np.concatenate(
            (account["natural_states"][5, 0], account["natural_states"][6, 1])
        )
        account["qualified_normalizers"] = np.r_[
            account["natural_normalizer"][5, 0, :, 0],
            account["natural_normalizer"][6, 1, :, 0],
        ]
        H.write_npz(out / "accounts" / f"{group_index:02d}.npz", **account)
    if not torch.backends.mps.is_available():
        raise RuntimeError("No undeclared projection backend")
    weight = OLD.weights().to("mps")
    for group_index, group in enumerate(GROUPS):
        if time.monotonic() - start > manifest["analysis_seconds_cap"]:
            raise TimeoutError("Original300second analysis cap; no extension")
        a = arrays(out / "accounts" / f"{group_index:02d}.npz")
        with torch.inference_mode():
            proj = (
                (
                    torch.tensor(
                        np.r_[a["qualified_fields"], a["reference_state"][None]],
                        dtype=torch.float32,
                        device="mps",
                    )
                    @ weight.T
                )
                .cpu()
                .numpy()
                .astype(float)
            )
        q = np.exp(proj[6] - logsumexp(proj[6]))
        if np.any(q == 0):
            raise ArithmeticError("Reference underflow; no support repair")
        means = proj[:6] @ q
        s = proj[:6] - means[:, None]
        gram = (s * q[None]) @ s.T
        H.write_npz(
            out / "accounts" / f"{group_index:02d}_gram.npz",
            gram=gram,
            means=means,
            normalizer_coefficients=a["qualified_normalizers"],
            calibration=means - a["qualified_normalizers"],
            reference_min=np.asarray(q.min()),
        )
        for axis in range(2):
            field_rows.append(
                {
                    "group": group_index,
                    "bank": group[0],
                    "condition": group[1],
                    "order": group[2],
                    "axis": M.AXES[axis],
                    "cross_wording_support": M.cross_wording_support(
                        gram[3 * axis : 3 * axis + 3, 3 * axis : 3 * axis + 3]
                    ),
                    "mean_field_energy": float(
                        np.trace(gram[3 * axis : 3 * axis + 3, 3 * axis : 3 * axis + 3])
                        / 3
                    ),
                }
            )
    own = [
        r
        for r in coefficients
        if r["mode"] == "natural" and r["mask"] == (5 if r["axis"] == M.AXES[0] else 6)
    ]
    mass = np.concatenate(
        [cubes[group, "natural"]["answer_mass"].ravel() for group in GROUPS]
    )
    self_error = max(
        float(
            np.max(
                abs(a["selected_logits"] - captured["natural", rid]["selected_logits"])
            )
        )
        for (mode, rid), a in captured.items()
        if mode == "self"
    )
    summary = {
        "packet": "HM155",
        "records": 32,
        "batches": 164,
        "inquiry_sequences": 1968,
        "qualified_scalar_positive": sum(r["logodds"] > 0 for r in own),
        "qualified_scalar_count": len(own),
        "qualified_scalar_rows": own,
        "actions": action_rows,
        "composition": composition_rows,
        "fields": field_rows,
        "answer_mass_mean": float(mass.mean()),
        "answer_mass_min": float(mass.min()),
        "own_donor_selected_error": self_error,
        "analysis_seconds": time.monotonic() - start,
        "human_personality_accuracy": False,
        "writer_reader_laws_identified_as_equal": False,
        "original_scope_complete": False,
    }
    for name, table in (
        ("characters", coefficients),
        ("actions", action_rows),
        ("composition", composition_rows),
        ("fields", field_rows),
    ):
        H.write_csv(out / (name + ".csv"), table)
    H.write_json(out / "summary.json", summary)
    H.write_json(
        out / "output_seal.json",
        {
            str(p.relative_to(out)): H.digest(p)
            for p in sorted(
                list((out / "accounts").glob("*"))
                + list(out.glob("*.csv"))
                + [out / "summary.json"]
            )
        },
    )
    J.status(
        out,
        "analysis",
        "COMPLETE",
        seconds=time.monotonic() - start,
        original_scope_complete=False,
    )
    print(
        {
            "qualified_positive": summary["qualified_scalar_positive"],
            "of": len(own),
            "self_error": self_error,
            "seconds": time.monotonic() - start,
        },
        flush=True,
    )


def main():
    """Expose separate prepare, collect and analyze stages with no auto reruns."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "collect", "analyze"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
