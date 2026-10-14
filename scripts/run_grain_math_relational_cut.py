#!/usr/bin/env python3
"""HM50: complete body/query cuts and owner-marker controls on the frozen reader."""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_relational_cut as M
from grain_research.vector_mechanisms.direct_traits import native_readout
from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from scripts.run_grain_math_construct_routing import save, write
from scripts.run_grain_math_evaluative_routing import load_arrays
from scripts.run_grain_math_relational_carrier import CODE as PARENT_CODE
from scripts.run_grain_math_relational_carrier import check as parent_check
from scripts.run_grain_math_relational_carrier import summarize
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_math_span_binding import hidden_tensor, load_observer
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_relational_cut_v1.json"
OUT = ROOT / "results/grain_math_response_v1/relational_cut_r1"
CODE = tuple(
    dict.fromkeys(
        (
            "configs/grain_math_relational_cut_v1.json",
            "docs/GRAIN_RELATIONAL_CUT_PLAN_V1.md",
            "grain_research/construction/math_relational_cut.py",
            "scripts/run_grain_math_relational_cut.py",
            "scripts/verify_grain_math_relational_cut.py",
            "tests/test_grain_math_relational_cut.py",
            *PARENT_CODE,
        )
    )
)


def prepare(out):
    """Freeze all48 exact-input cuts and7arms per receiver before fresh neural calls."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve existing versions")
    spec = read_json(SPEC)
    parent = ROOT / spec["parent"]
    prior = parent_check(parent)
    if (
        read_json(parent / "verification.json")["status"]
        != "VERIFIED_FINITE_HM49_NOT_GLOBAL"
        or spec["layer"] != prior["spec"]["layer"]
        or spec["targets"] != prior["spec"]["targets"]
        or spec["arms"] != list(M.ARMS)
    ):
        raise ValueError("Require verified HM49 and unchanged observer/scope")
    rows = {r["unit"]: r for r in read_json(parent / "00/receivers.json")}
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    partitions = {
        u: M.partition(
            r,
            tokenizer(
                r["rendered"], add_special_tokens=False, return_offsets_mapping=True
            ),
        )
        for u, r in rows.items()
    }
    pairs = {e["pair"]: e for e in read_json(parent / "00/design.json")}
    receivers = []
    for pair, old in sorted(pairs.items()):
        a, b = rows[old["receiver"]], rows[old["donor"]]
        proof = M.prove_pair(a, b, partitions[a["unit"]], partitions[b["unit"]])
        receivers.append(
            {
                k: old[k]
                for k in (
                    "pair",
                    "role",
                    "content",
                    "person",
                    "direction",
                    "receiver",
                    "donor",
                )
            }
            | {"partition": proof}
        )
    design = []
    for role in ("source", "application"):
        selected = [r for r in receivers if r["role"] == role]
        for r in selected:
            design.append(
                {**r, "kind": "capture", "arm": "native", "unit": len(design)}
            )
        for r in selected:
            for arm in spec["arms"]:
                design.append({**r, "kind": "patch", "arm": arm, "unit": len(design)})
    if (
        len(rows) != spec["expected_receivers"]
        or len(design) != spec["max_forwards"]
        or sum(r["role"] == "source" for r in receivers)
        != spec["expected_source_receivers"]
    ):
        raise ValueError("Incomplete48receiver/384call design")
    for directory in ("00/code", "01/units", "03"):
        (out / directory).mkdir(parents=True, exist_ok=True)
    write(out / "00/design.json", design)
    write(out / "00/receivers.json", list(rows.values()))
    code = {}
    for name in CODE:
        path = out / "00/code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
        code[name] = digest(path)
    parent_files = {
        name: digest(parent / name)
        for name in (
            "00/manifest.json",
            "00/design.json",
            "00/receivers.json",
            "verification.json",
            "01/observer.json",
        )
    }
    old = ROOT / prior["spec"]["parent"]
    historical = {}
    for unit in rows:
        for suffix in ("json", "npz"):
            path = old / f"01/units/unit_{unit:04d}.{suffix}"
            historical[str(path.relative_to(ROOT))] = digest(path)
    write(
        out / "00/manifest.json",
        {
            "packet": "HM50",
            "spec": spec,
            "alphabet": prior["alphabet"],
            "code": code,
            "assets": {
                name: digest(out / name)
                for name in ("00/design.json", "00/receivers.json")
            },
            "parent_files": parent_files,
            "historical_files": historical,
            "historical_root": str(old.relative_to(ROOT)),
            "model_config_sha256": digest(MODEL / "config.json"),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "known_material_followup": True,
            "model_weights_not_separately_hashed": True,
            "human_truth": False,
        },
    )
    write(
        out / "run_status.json",
        {
            "status": "PREPARED",
            "forward_calls": 0,
            "attempted_forward_calls": 0,
            "completed_units": 0,
            "model_seconds": 0.0,
            "in_flight": None,
        },
    )
    print(
        "PREPARED:48 complete cuts/384 fixed calls; only2name tokens differ", flush=True
    )


def check(out):
    """Bind live/frozen code, token proofs, selected old states and local model config."""
    m = read_json(out / "00/manifest.json")
    for name, sha in m["code"].items():
        if digest(ROOT / name) != sha or digest(out / "00/code" / name) != sha:
            raise ValueError("Changed frozen code " + name)
    for name, sha in m["assets"].items():
        if digest(out / name) != sha:
            raise ValueError("Changed fixed material " + name)
    for name, sha in m["parent_files"].items():
        if digest(ROOT / m["spec"]["parent"] / name) != sha:
            raise ValueError("Changed parent " + name)
    for name, sha in m["historical_files"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Changed historical score/capture " + name)
    if digest(MODEL / "config.json") != m["model_config_sha256"]:
        raise ValueError("Changed local model config")
    return m


def observe_full(observer, row, alphabet, layer, patch=None):
    """Capture every pre/post-patch token at one cut, continuing upper layers freely."""
    import torch

    ids = torch.tensor(
        [row["input_ids"]],
        dtype=torch.long,
        device=next(observer.model.parameters()).device,
    )
    captured = {}

    def hook(_module, _args, output):
        h = hidden_tensor(output)
        if patch is not None:
            positions, values = patch
            if np.shape(values) != (len(positions), h.shape[-1]):
                raise ValueError("Complete-cut replacement shape mismatch")
            captured["before"] = h[0].detach().cpu().numpy().copy()
            h = h.clone()
            h[0, positions] = torch.tensor(values, dtype=h.dtype, device=h.device)
        captured["full"] = h[0].detach().cpu().numpy().copy()
        return (h, *output[1:]) if isinstance(output, tuple) else h

    handle = observer.model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            output = observer.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
        observer.calls += 1
        observer.input_tokens += len(row["input_ids"])
        observer.processed_tokens += len(row["input_ids"])
        torch.mps.synchronize()
        return native_readout(
            output.logits[0, -1].float().cpu().numpy(), alphabet
        ), captured
    finally:
        handle.remove()


def collect(out):
    """Run source192 then application192, committing each attempted call before execution."""
    m, status = check(out), read_json(out / "run_status.json")
    if status["status"] == "COLLECTED":
        return
    if status["in_flight"] is not None or status["status"] == "STOPPED":
        raise RuntimeError(
            "Stopped/uncertain calls require inspection, not automatic retry"
        )
    spec = m["spec"]
    rows = {r["unit"]: r for r in read_json(out / "00/receivers.json")}
    design = read_json(out / "00/design.json")
    capture_units = {e["receiver"]: e["unit"] for e in design if e["kind"] == "capture"}
    states = {}
    started, previous = time.monotonic(), status["model_seconds"]
    try:
        observer = load_observer()
        actual, prior = (
            observer.manifest(),
            read_json(ROOT / spec["parent"] / "01/observer.json"),
        )
        for key in (
            "model",
            "torch",
            "transformers",
            "dtype",
            "device",
            "hidden_size",
            "layers",
            "max_tokens",
            "attention",
            "training",
            "template_sha256",
        ):
            if actual[key] != prior[key]:
                raise ValueError("Observer drift " + key)
        write(out / "01/observer.json", actual)
        for role in ("source", "application"):
            if role == "application":
                barrier = read_json(out / "01/source_barrier.json")
                for name, sha in barrier["files"].items():
                    if digest(out / name) != sha:
                        raise ValueError("Changed sealed source output")
            for entry in design:
                if entry["role"] != role:
                    continue
                path = out / f"01/units/unit_{entry['unit']:04d}.json"
                if path.exists():
                    _, arrays = load_arrays(path)
                    if entry["kind"] == "capture":
                        states[entry["receiver"]] = arrays["full"]
                    continue
                elapsed = previous + time.monotonic() - started
                if not M.call_allowed(status, spec, elapsed):
                    raise TimeoutError("Finite384call/600second contract reached")
                patch = None
                if entry["kind"] == "patch":
                    for unit in (entry["receiver"], entry["donor"]):
                        if unit not in states:
                            _, arrays = load_arrays(
                                out / f"01/units/unit_{capture_units[unit]:04d}.json"
                            )
                            states[unit] = arrays["full"]
                    patch = M.patch_values(
                        entry["partition"],
                        states[entry["receiver"]],
                        states[entry["donor"]],
                        entry["arm"],
                        entry["pair"],
                        spec["random_seed"],
                    )
                status.update(
                    status="COLLECTING", in_flight=entry["unit"], model_seconds=elapsed
                )
                status["attempted_forward_calls"] += 1
                write(out / "run_status.json", status)
                native, captured = observe_full(
                    observer,
                    rows[entry["receiver"]],
                    m["alphabet"],
                    spec["layer"],
                    patch,
                )
                status["forward_calls"] += 1
                if patch is not None:
                    captured.update(
                        patch_positions=np.asarray(patch[0], int), patch_values=patch[1]
                    )
                else:
                    states[entry["receiver"]] = captured["full"]
                save(
                    path,
                    {
                        k: entry[k]
                        for k in (
                            "unit",
                            "pair",
                            "role",
                            "kind",
                            "arm",
                            "receiver",
                            "donor",
                        )
                    }
                    | {"native": native},
                    captured,
                )
                status.update(
                    in_flight=None,
                    completed_units=status["completed_units"] + 1,
                    model_seconds=previous + time.monotonic() - started,
                )
                write(out / "run_status.json", status)
                if status["completed_units"] % 24 == 0:
                    print(
                        f"{role}: {status['completed_units']}/384 actual calls",
                        flush=True,
                    )
            if role == "source":
                paths = [
                    out / f"01/units/unit_{e['unit']:04d}.json"
                    for e in design
                    if e["role"] == role
                ]
                write(
                    out / "01/source_barrier.json",
                    {
                        "source_calls": len(paths),
                        "actual_calls_at_seal": status["forward_calls"],
                        "created_utc": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                        "files": {
                            str(p.relative_to(out)): digest(p)
                            for path in paths
                            for p in (path, path.with_suffix(".npz"))
                        },
                    },
                )
        if (
            any(
                status[k] != spec["max_forwards"]
                for k in ("forward_calls", "attempted_forward_calls", "completed_units")
            )
            or status["model_seconds"] > spec["max_model_seconds"]
        ):
            raise ValueError("Incomplete or over-budget collection")
        status["status"] = "COLLECTED"
        write(out / "run_status.json", status)
    except BaseException as error:
        status.update(
            status="STOPPED",
            error=repr(error),
            model_seconds=previous + time.monotonic() - started,
        )
        write(out / "run_status.json", status)
        raise


def analyze(out):
    """Report signed actions, exact four cells and all squared/cross-energy terms."""
    m = check(out)
    if read_json(out / "run_status.json")["status"] != "COLLECTED":
        raise ValueError("Actual complete collection required")
    design = read_json(out / "00/design.json")
    data = {
        e["unit"]: load_arrays(out / f"01/units/unit_{e['unit']:04d}.json")[0]
        for e in design
    }
    capture_units = {e["receiver"]: e["unit"] for e in design if e["kind"] == "capture"}
    scores = {u: meta["native"]["conditional_score"] for u, meta in data.items()}
    by_pair, effects, replay = defaultdict(dict), [], []
    for e in design:
        native, donor = (
            scores[capture_units[e["receiver"]]],
            scores[capture_units[e["donor"]]],
        )
        if e["kind"] == "capture":
            old = read_json(
                ROOT / m["historical_root"] / f"01/units/unit_{e['receiver']:04d}.json"
            )["native"]["conditional_score"]
            replay.append(
                {
                    "pair": e["pair"],
                    "role": e["role"],
                    "native": native,
                    "historical": old,
                    "difference": native - old,
                }
            )
            continue
        patched = scores[e["unit"]]
        by_pair[e["pair"]][e["arm"]] = patched
        effects.append(
            {
                k: e[k]
                for k in (
                    "unit",
                    "pair",
                    "role",
                    "content",
                    "person",
                    "direction",
                    "arm",
                )
            }
            | {
                "native": native,
                "donor": donor,
                "patched": patched,
                "candidate_mass": data[e["unit"]]["native"]["candidate_mass"],
                "full_change": donor - native,
                "patch_change": patched - native,
                "remaining": donor - patched,
            }
        )
    cells = []
    for e in design:
        if e["kind"] != "capture":
            continue
        values = by_pair[e["pair"]]
        native, donor = scores[e["unit"]], scores[capture_units[e["donor"]]]
        _, a = load_arrays(out / f"01/units/unit_{e['unit']:04d}.json")
        _, b = load_arrays(out / f"01/units/unit_{capture_units[e['donor']]:04d}.json")
        prefix_error = float(
            np.max(
                abs(
                    a["full"][e["partition"]["prefix"]].astype(float)
                    - b["full"][e["partition"]["prefix"]].astype(float)
                )
            )
        )
        cells.append(
            {k: e[k] for k in ("pair", "role", "content", "person", "direction")}
            | M.four_cells(native, values["body"], values["suffix"], values["full"])
            | {
                "fresh_donor": donor,
                "full_replay_error": values["full"] - donor,
                "noop_error": values["noop"] - native,
                "common_prefix_max_difference": prefix_error,
            }
        )
    groups, content_groups, cell_groups = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    for r in effects:
        groups[r["role"], r["arm"]].append(r)
        content_groups[r["role"], r["content"], r["arm"]].append(r)
    for r in cells:
        cell_groups[r["role"]].append(r)
    energy = []
    for role, values in cell_groups.items():
        x = np.asarray(
            [
                [r[k] for k in ("body_action", "suffix_action", "interaction")]
                for r in values
            ]
        )
        terms = {
            "body_square": float(np.mean(x[:, 0] ** 2)),
            "suffix_square": float(np.mean(x[:, 1] ** 2)),
            "interaction_square": float(np.mean(x[:, 2] ** 2)),
            "body_suffix_cross": float(2 * np.mean(x[:, 0] * x[:, 1])),
            "body_interaction_cross": float(2 * np.mean(x[:, 0] * x[:, 2])),
            "suffix_interaction_cross": float(2 * np.mean(x[:, 1] * x[:, 2])),
        }
        energy.append(
            {
                "role": role,
                "rows": len(values),
                **terms,
                "joint_square": float(
                    np.mean([r["joint_action"] ** 2 for r in values])
                ),
                "reconstructed_square": sum(terms.values()),
            }
        )
    for name, values in (
        ("effects.csv", effects),
        ("four_cells.csv", cells),
        ("native_replay.csv", replay),
        ("signed_energy.csv", energy),
    ):
        write_csv(out / "03" / name, values)
    summaries = [{"role": k[0], "arm": k[1], **summarize(v)} for k, v in groups.items()]
    write_csv(out / "03/arm_summary.csv", summaries)
    write_csv(
        out / "03/content_summary.csv",
        [
            {"role": k[0], "content": k[1], "arm": k[2], **summarize(v)}
            for k, v in content_groups.items()
        ],
    )
    write(
        out / "03/summary.json",
        {
            "packet": "HM50",
            "effects": len(effects),
            "four_cells": len(cells),
            "groups": {"/".join(k): summarize(v) for k, v in groups.items()},
            "full_replay_max_error": max(abs(r["full_replay_error"]) for r in cells),
            "noop_max_error": max(abs(r["noop_error"]) for r in cells),
            "native_historical_max_error": max(abs(r["difference"]) for r in replay),
            "prefix_max_difference": max(
                r["common_prefix_max_difference"] for r in cells
            ),
            "interaction_rms": {
                k: float(np.sqrt(np.mean([r["interaction"] ** 2 for r in v])))
                for k, v in cell_groups.items()
            },
            "known_material_followup_not_blind_replication": True,
            "human_personality_accuracy": False,
            "original_scope_complete": False,
        },
    )
    paths = (
        list((out / "01").rglob("*.json"))
        + list((out / "01").rglob("*.npz"))
        + list((out / "03").glob("*.csv"))
        + [out / "03/summary.json"]
    )
    write(
        out / "03/artifact_hashes.json",
        {str(p.relative_to(out)): digest(p) for p in paths},
    )
    print("ANALYZED:336effects/48fourcells/all signed cross energies", flush=True)


def main():
    """Expose finite explicit stages; do not restart stopped acquisition implicitly."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "collect", "analyze"))
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    with collector_guard(args.output):
        {"prepare": prepare, "collect": collect, "analyze": analyze}[args.stage](
            args.output
        )


if __name__ == "__main__":
    main()
