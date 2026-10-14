#!/usr/bin/env python3
"""Bounded public-text causal route experiment, not a prediction accuracy search."""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_history_routing as M
from grain_research.vector_mechanisms.navigation import MODEL, write_json
from grain_research.vector_mechanisms.prefix import capture_prefix
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_observer_parity_hm96 import load, seal, sha, table, verify, write
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_history_routing_hm101.json"
CODE = (
    "configs/grain_history_routing_hm101.json",
    "docs/GRAIN_HISTORY_ROUTING_PLAN_HM101.md",
    "grain_research/construction/math_history_routing.py",
    "scripts/run_grain_history_routing_hm101.py",
    "scripts/verify_grain_history_routing_hm101.py",
    "tests/test_grain_math_history_routing.py",
    "grain_research/vector_mechanisms/prefix.py",
    "grain_research/construction/math_cold_evidence.py",
    "grain_research/vector_mechanisms/navigation.py",
    "grain_research/vector_mechanisms/trace.py",
    "scripts/run_grain_math_span_binding.py",
    "scripts/run_grain_upstream_routes.py",
    "scripts/run_grain_observer_parity_hm96.py",
)
ARM_NAMES = ("neither", "direct_only", "relay_only", "both")
COMPONENTS = ("baseline", "direct", "relay", "interaction")


def settings():
    """Resolve the sole fixed protocol and output destination."""
    spec = load(SPEC)
    return spec, ROOT / spec["output"]


def check(out):
    """Reject changed inputs, source calibration or intervention code."""
    manifest = load(out / "00/manifest.json")
    verify(ROOT, manifest["code"] | manifest["source"])
    verify(out, manifest["inputs"])
    return manifest


def prepare(out):
    """Freeze paired current tokens and complete intervention support before calls."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve prior and partial packets")
    spec, _ = settings()
    source = ROOT / spec["source"]
    old = load(source / "00/manifest.json")
    verify(ROOT, old["code"] | old["source"])
    verify(source, old["inputs"])
    if (
        load(source / "verification.json")["status"]
        != "VERIFIED_FINITE_HM97_NOT_GLOBAL"
    ):
        raise ValueError("Verified finite source calibration required")
    bundle = load(source / "00/bundle.json")
    if bundle["targets"] != spec["targets"]:
        raise ValueError("Source target battery changed")
    materials = [
        load(ROOT / spec[n]) for n in ("source_material", "application_material")
    ]
    banks = [
        {
            t: materials[0]["behaviours"][t][spec["source_lexicon"]]
            for t in spec["targets"]
        },
        materials[1]["behaviours"],
    ]
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    system = load(ROOT / spec["system_material"])["system"]
    marker = "__GRAIN_BODY__"
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": marker}],
        tokenize=False,
        add_generation_prompt=False,
    )
    prefix = tokenizer.encode(rendered.split(marker)[0], add_special_tokens=False)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    history_tokens = [
        {
            t: [encode(f"On occasion K1, Person A {word}.") for word in bank[t]]
            for t in spec["targets"]
        }
        for bank in banks
    ]
    width = max(
        len(h) for bank in history_tokens for family in bank.values() for h in family
    )
    pad = tokenizer.pad_token_id
    if pad is None:
        raise ValueError("Frozen tokenizer has no ignored pad ID")
    rows, controls = [], []
    for corpus, bridge, family, i, j in itertools.product(
        range(2), range(2), range(6), range(3), range(3)
    ):
        target = spec["targets"][family]
        h = history_tokens[corpus][target][i]
        b = encode("\n" + spec["bridges"][bridge] + "\nOn occasion K2, Person A")
        c = encode(" " + banks[corpus][target][j] + ".")
        history = list(range(len(prefix), len(prefix) + width))
        padding = history[len(h) :]
        current = list(
            range(len(prefix) + width + len(b), len(prefix) + width + len(b) + len(c))
        )
        ids = prefix + h + [pad] * len(padding) + b + c
        row = {
            "body": f"On occasion K1, Person A {banks[corpus][target][i]}.\n{spec['bridges'][bridge]}\nOn occasion K2, Person A {banks[corpus][target][j]}.",
            "corpus": corpus,
            "bridge": bridge,
            "family": family,
            "previous": i,
            "current_state": j,
            "input_ids": ids,
            "history": history,
            "padding": padding,
            "current": current,
            "relay": [p for p in range(len(ids)) if p not in set(history + current)],
        }
        for arm, (direct, relay) in enumerate(spec["arms"]):
            mask = M.route_mask(
                len(ids), history, row["relay"], current, padding, direct, relay
            )
            reach = M.dependency_support(mask, history, 17)[current]
            if bool(reach.any()) != bool(direct or relay):
                raise ValueError("Incorrect intervention dependency graph")
            rows.append(
                {
                    **row,
                    "unit": len(rows),
                    "arm": arm,
                    "direct": direct,
                    "indirect": relay,
                }
            )
    identity_checks = 0
    grouped = {}
    for row in rows:
        key = row["corpus"], row["bridge"], row["family"], row["current_state"]
        h = set(row["history"])
        fixed = (
            row["current"],
            [v for p, v in enumerate(row["input_ids"]) if p not in h],
        )
        if key in grouped and grouped[key] != fixed:
            raise ValueError(
                "Past manipulation changed current position or non-history tokens"
            )
        grouped[key] = fixed
        identity_checks += 1
    first = rows[0]
    keep = [p for p in range(len(first["input_ids"])) if p not in first["padding"]]
    back = {p: k for k, p in enumerate(keep)}
    control = {
        **first,
        "input_ids": [first["input_ids"][p] for p in keep],
        "padding": [],
    }
    for name in ("history", "relay", "current"):
        control[name] = [back[p] for p in first[name] if p in back]
    controls.append(control)
    if len(rows) + 2 != spec["max_forwards"]:
        raise ValueError("Unexpected finite forward count")
    (out / "00").mkdir(parents=True)
    write(out / "00/rows.json", rows)
    write(out / "00/controls.json", controls)
    write(out / "00/bundle.json", bundle)
    source_files = [
        source / n for n in ("00/manifest.json", "00/bundle.json", "verification.json")
    ]
    source_files.extend(
        ROOT / spec[n]
        for n in ("source_material", "application_material", "system_material")
    )
    write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "code": {n: sha(ROOT / n) for n in CODE},
            "source": {str(p.relative_to(ROOT)): sha(p) for p in source_files},
            "model": old["model"],
            "inputs": seal(out, out / "00"),
            "paired_token_position_checks": identity_checks,
            "history_width": width,
            "new_token_streams": len(rows) // 4,
            "no_target_query_answer_fit": True,
            "padding_is_protocol_not_natural_text": True,
            "original_scope_complete": False,
        },
    )
    print(
        "PREPARED HM101",
        len(rows) // 4,
        "bodies",
        len(rows) + 2,
        "forwards",
        flush=True,
    )


def unit(out, index):
    """Load a completed unit only after its primitive bytes and trace agree."""
    path = out / f"01/units/unit_{index:04d}"
    meta = load(path.with_suffix(".json"))
    if sha(path.with_suffix(".npz")) != meta["sha256"]:
        raise ValueError("Partial or changed primitive")
    trace = meta["trace"]
    if trace["completed_blocks"] != list(range(17)) or trace["forbidden_calls"]:
        raise ValueError("Illegal model execution trace")
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
        return {n: z[n] for n in z.files}, meta


def collect(out):
    """Execute the fixed masks once per unit, with resumable output and time cap."""
    manifest = check(out)
    rows = load(out / "00/rows.json")
    verify(MODEL, manifest["model"])
    if (out / "01/raw_seal.json").exists():
        raise FileExistsError("Completed collection must not be rerun")
    (out / "01/units").mkdir(parents=True, exist_ok=True)
    status_path = out / "01/status.json"
    status = (
        load(status_path)
        if status_path.exists()
        else {"state": "RUNNING", "completed": 0, "model_seconds": 0.0}
    )
    observer = load_observer()
    with collector_guard(out):
        for row in rows:
            path = out / f"01/units/unit_{row['unit']:04d}"
            if path.with_suffix(".json").exists():
                unit(out, row["unit"])
                continue
            if status["model_seconds"] >= manifest["spec"]["max_model_seconds"]:
                raise TimeoutError(
                    "Finite model time budget reached; preserve checkpoint"
                )
            begin = time.monotonic()
            raw, trace = M.capture_routed_prefix(
                observer, row, row["direct"], row["indirect"]
            )
            elapsed = time.monotonic() - begin
            with path.with_suffix(".npz").open("xb") as stream:
                np.savez_compressed(stream, **raw)
            write(
                path.with_suffix(".json"),
                {
                    "unit": row["unit"],
                    "trace": trace,
                    "seconds": elapsed,
                    "sha256": sha(path.with_suffix(".npz")),
                },
            )
            status = {
                "state": "RUNNING",
                "completed": row["unit"] + 1,
                "model_seconds": status["model_seconds"] + elapsed,
            }
            write_json(status_path, status)
            if status["completed"] % 72 == 0:
                print("HM101", status, flush=True)
        control_path = out / "01/noop_control.npz"
        if not control_path.exists():
            row = load(out / "00/controls.json")[0]
            begin = time.monotonic()
            native, trace = capture_prefix(observer, row["input_ids"])
            routed, hook_trace = M.capture_routed_prefix(observer, row, 1, 1)
            with control_path.open("xb") as stream:
                np.savez_compressed(
                    stream,
                    native_k=native["k"][row["current"]],
                    native_v=native["v"][row["current"]],
                    routed_k=routed["k"],
                    routed_v=routed["v"],
                )
            write(
                out / "01/noop_control.json",
                {
                    "native_trace": trace,
                    "hook_trace": hook_trace,
                    "seconds": time.monotonic() - begin,
                },
            )
            status["model_seconds"] += load(out / "01/noop_control.json")["seconds"]
        status.update(state="COLLECTED", actual_forwards=len(rows) + 2)
        if status["model_seconds"] > manifest["spec"]["max_model_seconds"]:
            raise TimeoutError(
                "Completed scientific capture exceeds registered model budget"
            )
        write_json(status_path, status)
        write(
            out / "01/raw_seal.json",
            seal(out, out / "01/units")
            | {
                str(p.relative_to(out)): sha(p)
                for p in (control_path, out / "01/noop_control.json")
            },
        )
    print("COLLECTED HM101", status, flush=True)


def analyze(out):
    """Resolve vector writing, target readout and signed author route interactions."""
    manifest = check(out)
    verify(out, load(out / "01/raw_seal.json"))
    status = load(out / "01/status.json")
    if status["state"] != "COLLECTED":
        raise ValueError("Incomplete real prefix collection")
    spec, bundle = manifest["spec"], load(out / "00/bundle.json")
    fields = np.zeros((2, 2, 6, 3, 3, 4, 2, 6))
    vectors = np.zeros((2, 2, 6, 3, 3, 4, 256))
    no_history = {}
    invariance = 0.0
    for row in load(out / "00/rows.json"):
        raw, _ = unit(out, row["unit"])
        index = (
            row["corpus"],
            row["bridge"],
            row["family"],
            row["previous"],
            row["current_state"],
            row["arm"],
        )
        fields[index] = M.read_current(raw["k"], raw["v"], max(row["current"]), bundle)
        vectors[index] = raw["v"].mean(0).ravel()
        if row["arm"] == 0:
            key = row["corpus"], row["bridge"], row["family"], row["current_state"]
            if key in no_history:
                invariance = max(
                    invariance,
                    *(
                        float(np.max(np.abs(raw[n] - no_history[key][n])))
                        for n in ("k", "v")
                    ),
                )
            else:
                no_history[key] = raw
    with np.load(out / "01/noop_control.npz", allow_pickle=False) as z:
        noop = max(
            float(np.max(np.abs(z[f"native_{n}"] - z[f"routed_{n}"])))
            for n in ("k", "v")
        )
    if max(invariance, noop) > spec["carrier_atol"]:
        write(
            out / "control_failure.json",
            {
                "status": "CONTROL_FAILED",
                "all_cut_max": invariance,
                "noop_max": noop,
                "tolerance": spec["carrier_atol"],
            },
        )
        raise ValueError("Observation control failed; do not interpret route effects")
    dest = out / "02"
    dest.mkdir()
    contrasts = np.empty((2, 2, 6, 4, 2, 6))
    components = np.empty_like(contrasts)
    author_rows, vector_rows = [], []
    recon = 0.0
    for corpus, bridge, family in itertools.product(range(2), range(2), range(6)):
        base = (corpus, bridge, family)
        corners = fields[base].transpose(2, 0, 1, 3, 4)
        contrasts[base] = np.stack([M.author_contrast(f) for f in corners])
        components[base] = M.mobius(contrasts[base])
        recon = max(
            recon, float(np.max(np.abs(components[base].sum(0) - contrasts[base][3])))
        )
        for component, field, target in itertools.product(range(4), range(2), range(6)):
            author_rows.append(
                {
                    "corpus": corpus,
                    "bridge": bridge,
                    "family": spec["targets"][family],
                    "target": spec["targets"][target],
                    "field": ("uniform_value", "local_query")[field],
                    "component": COMPONENTS[component],
                    "contrast": float(components[base][component, field, target]),
                    "raw_arm_contrast": float(
                        contrasts[base][component, field, target]
                    ),
                }
            )
        vc = vectors[base].transpose(2, 0, 1, 3)
        vm = M.mobius(vc)
        # Remove the same actual j->j reference independently in every arm.
        history = vm - np.stack([vm[:, j, j] for j in range(3)], axis=1)[:, None]
        total = history[1:].sum(0)
        denom = float(np.sum(total**2))
        for c in range(1, 4):
            vector_rows.append(
                {
                    "corpus": corpus,
                    "bridge": bridge,
                    "family": spec["targets"][family],
                    "component": COMPONENTS[c],
                    "history_vector_rms": float(np.sqrt(np.mean(history[c] ** 2))),
                    "signed_alignment": float(np.sum(history[c] * total) / denom)
                    if denom
                    else 0.0,
                    "energy_ratio": float(np.sum(history[c] ** 2) / denom)
                    if denom
                    else 0.0,
                }
            )
    if recon > spec["arithmetic_atol"]:
        raise ValueError("Finite author-route account does not close")
    with (dest / "accounts.npz").open("xb") as stream:
        np.savez_compressed(
            stream,
            fields=fields,
            vectors=vectors,
            contrasts=contrasts,
            components=components,
        )
    table(dest / "author_contrasts.csv", author_rows)
    table(dest / "history_vectors.csv", vector_rows)
    summary = {
        "status": "ANALYZED_FINITE_HM101_NOT_GLOBAL",
        "actual_forwards": status["actual_forwards"],
        "model_seconds": status["model_seconds"],
        "new_token_streams": manifest["new_token_streams"],
        "all_cut_current_kv_max": invariance,
        "noop_hook_kv_max": noop,
        "author_route_reconstruction_error": recon,
        "max_abs_author_contrast": float(np.max(np.abs(contrasts))),
        "max_abs_author_interaction": float(np.max(np.abs(components[..., 3, :, :]))),
        "not_personality_accuracy_or_information_fraction": True,
        "new_fits": 0,
        "original_scope_complete": False,
    }
    write(dest / "summary.json", summary)
    write(out / "02/output_seal.json", seal(out, dest))
    print("ANALYZED HM101", summary, flush=True)


def main():
    """Execute only the explicitly requested finite stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "collect", "analyze", "verify"))
    args = parser.parse_args()
    _, out = settings()
    if args.stage == "verify":
        from scripts.verify_grain_history_routing_hm101 import run

        run(out)
    else:
        globals()[args.stage](out)


if __name__ == "__main__":
    main()
