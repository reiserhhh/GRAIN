#!/usr/bin/env python3
"""HM31 bounded identical-H/different-c decoder recombination experiment."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_context_fiber as F
from grain_research.construction import math_crossed_field as C
from grain_research.construction import math_position_support as P
from grain_research.construction import math_source_field as S
from grain_research.vector_mechanisms.direct_traits import native_readout
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_crossed_field import fields
from scripts.run_grain_math_position_support import (
    array_unit,
    baseline_path,
    check,
    forward,
    hidden_tensor,
    load_observer,
    now,
    patch_path,
    recover_in_flight,
    save_array_unit,
)
from scripts.run_grain_math_response import write_csv, write_json
from scripts.run_grain_upstream_routes import collector_guard

OUT = ROOT / "results/grain_math_response_v1/context_fiber_recombination_r1"
SPEC = ROOT / "configs/grain_math_context_fiber_v1.json"
PLAN = ROOT / "docs/GRAIN_CONTEXT_FIBER_RECOMBINATION_PLAN_V1.md"
CODE = (
    "grain_research/construction/math_context_fiber.py",
    "scripts/run_grain_math_context_fiber.py",
    "scripts/verify_grain_math_context_fiber.py",
    "tests/test_grain_math_context_fiber.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)


def prepare(out):
    """Freeze all192 corners and parent dependencies before96 new references."""
    if out.exists():
        raise FileExistsError("preserve existing experiment and resume stages")
    spec = read_json(SPEC)
    parent = ROOT / spec["parent"]
    pm = check(parent)
    verified = ROOT / spec["parent_verification"] / "summary.json"
    if read_json(verified)["status"] != "VERIFIED_FINITE_HM30_NOT_GLOBAL":
        raise ValueError("verified HM30 required")
    dependencies = dict(pm["files"])
    ledger = read_json(parent / "02/artifact_hashes.json")
    for name, sha in ledger.items():
        if digest(parent / name) != sha:
            raise ValueError("frozen HM30 output changed " + name)
        dependencies[str((parent / name).relative_to(ROOT))] = sha
    for path in (verified, parent / "02/artifact_hashes.json"):
        dependencies[str(path.relative_to(ROOT))] = digest(path)
    dependencies.update({name: digest(ROOT / name) for name in CODE})
    old = read_json(parent / "00/design.json")
    design = F.corners(old, spec["context_pairs"])
    if len(design) != 192 or sum(not r["diagonal"] for r in design) != 96:
        raise ValueError("complete48four-corner design required")
    if list(C.ARMS) != spec["arms"] or len(fields(parent)) != 15:
        raise ValueError("frozen arm/field inventory changed")
    for r in design:
        donor, receiver = old[r["donor_unit"]], old[r["receiver_unit"]]
        if (
            donor["spans"]["A_property"]["raw"]
            != receiver["spans"]["A_property"]["raw"]
        ):
            raise ValueError("same property bytes required")
    for name in ("00/code", "01/baselines", "01/patches", "02"):
        (out / name).mkdir(parents=True, exist_ok=True)
    write_json(out / "00/design.json", design)
    for name in ("directions.npz", "queries.json", "documents.json"):
        (out / "00" / name).write_bytes((parent / "00" / name).read_bytes())
    for name in CODE:
        dst = out / "00/code" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / name).read_bytes())
    write_json(
        out / "00/manifest.json",
        {
            "files": dependencies,
            "spec": spec,
            "alphabet": pm["alphabet"],
            "inputs": {
                "00/" + n: digest(out / "00" / n)
                for n in (
                    "design.json",
                    "directions.npz",
                    "queries.json",
                    "documents.json",
                )
            },
            "frozen_before_outputs": now(),
            "new_fits": 0,
            "context_includes_query_and_layout": True,
        },
    )
    write_json(
        out / "run_status.json",
        {
            "status": "PREPARED",
            "forward_calls": 0,
            "backward_calls": 0,
            "in_flight": None,
            "model_seconds": 0.0,
        },
    )
    print({"status": "PREPARED", "corners": 192, "new_references": 96}, flush=True)


def ref_path(out, row, m):
    """Resolve inherited diagonals without replaying their native calls."""
    return (
        baseline_path(ROOT / m["spec"]["parent"], row["donor_unit"])
        if row["diagonal"]
        else baseline_path(out, row["unit"])
    )


def end_path(out, row, m, arm, sign):
    """Resolve each endpoint's original acquisition address."""
    return (
        patch_path(ROOT / m["spec"]["parent"], row["donor_unit"], arm, sign)
        if row["diagonal"]
        else patch_path(out, row["unit"], arm, sign)
    )


def parent_state(parent, unit):
    """Read hash-bound natural property, gradient and query arrays."""
    path = baseline_path(parent, unit)
    array_unit(path)
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def observe(observer, row, m, receiver_h, replacement, gradient):
    """Differentiate after replacement, holding every receiver complement state fixed."""
    import torch

    slots = {}
    positions = row["spans"]["A_property"]["positions"]
    device = next(observer.model.parameters()).device

    def hook(_module, _args, output):
        original = hidden_tensor(output)
        changed, audit = F.replace_leaf(
            original, positions, receiver_h, replacement, gradient
        )
        slots.update(
            {
                "leaf": changed,
                "audit": audit,
                "h": changed[0, positions].detach().cpu().numpy().copy(),
                "q": changed[0, -1].detach().cpu().numpy().copy(),
            }
        )
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    handle = observer.model.model.layers[m["spec"]["layer"]].register_forward_hook(hook)
    try:
        if not gradient:
            native = forward(observer, row["input_ids"], m["alphabet"])
            return native, slots["audit"]
        ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
        result = observer.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )
        observer.calls += 1
        logits = result.logits[0, -1]
        values = torch.tensor(
            m["alphabet"]["values"], dtype=torch.float32, device=device
        )
        score = (
            torch.softmax(logits[m["alphabet"]["token_ids"]].float(), dim=0) * values
        ).sum()
        g = torch.autograd.grad(score, slots["leaf"])[0][0, positions]
        native = native_readout(logits.detach().float().cpu().numpy(), m["alphabet"])
        return (
            {
                "h": slots["h"],
                "q": slots["q"],
                "receiver_h": receiver_h,
                "gradient": g.detach().cpu().numpy().copy(),
            },
            native,
            float(score.detach().cpu()),
            slots["audit"],
        )
    finally:
        handle.remove()


def calls(out, source):
    """Acquire only off-diagonal references/endpoints, source then held."""
    m = check(out)
    spec = m["spec"]
    parent = ROOT / spec["parent"]
    fields(parent)
    status = recover_in_flight(out, read_json(out / "run_status.json"))
    barrier = out / "01/source_barrier.json"
    if source == barrier.exists():
        raise ValueError("source acquisition precedes checkpoint; held follows it")
    barrier_sha = None if source else digest(barrier)
    if not source:
        for name, sha in read_json(barrier)["source_artifacts"].items():
            if digest(out / name) != sha:
                raise ValueError("source checkpoint changed")
    design = read_json(out / "00/design.json")
    new = [r for r in design if not r["diagonal"]]
    rows = [r for r in new if (r["role"] == "source") == source]
    bases = sum(baseline_path(out, r["unit"]).exists() for r in new)
    ends = sum(
        patch_path(out, r["unit"], a, s).exists()
        for r in new
        for a in spec["arms"]
        for s in ("plus", "minus")
    )
    if (status["forward_calls"], status["backward_calls"]) != (bases + ends, bases):
        raise ValueError("actual-call ledger disagrees with committed new artifacts")
    if all(
        baseline_path(out, r["unit"]).exists()
        and all(
            patch_path(out, r["unit"], a, s).exists()
            for a in spec["arms"]
            for s in ("plus", "minus")
        )
        for r in rows
    ):
        print("ALREADY_ACQUIRED", flush=True)
        return
    started, elapsed = time.monotonic(), status["model_seconds"]
    if elapsed >= spec["maximum_model_seconds"]:
        raise ValueError("model time exhausted")
    observer = load_observer()
    if any(p.requires_grad for p in observer.model.parameters()):
        raise ValueError("fixed neural weights required")
    write_json(
        out / ("01/source_observer.json" if source else "01/held_observer.json"),
        observer.manifest(),
    )
    with np.load(out / "00/directions.npz", allow_pickle=False) as bank:
        directions = {q: C.directions(bank[q + "_b"]) for q in spec["queries"]}

    def begin(path, backwards):
        nonlocal status
        seconds = elapsed + time.monotonic() - started
        if (
            status["forward_calls"] >= spec["maximum_forwards"]
            or status["backward_calls"] + backwards > spec["maximum_backwards"]
            or seconds >= spec["maximum_model_seconds"]
        ):
            write_json(
                out / "run_status.json",
                {**status, "status": "RESOURCE_BOUNDARY", "model_seconds": seconds},
            )
            return False
        status = {
            **status,
            "status": "SOURCE_CALLS" if source else "HELD_CALLS",
            "model_seconds": seconds,
            "in_flight": {
                "file": str(path.relative_to(out)),
                "backwards": backwards,
                "started_at": now(),
            },
        }
        write_json(out / "run_status.json", status)
        return True

    def finish(backwards):
        nonlocal status
        status = {
            **status,
            "in_flight": None,
            "forward_calls": status["forward_calls"] + 1,
            "backward_calls": status["backward_calls"] + backwards,
            "model_seconds": elapsed + time.monotonic() - started,
        }
        write_json(out / "run_status.json", status)

    try:
        for r in rows:
            donor = parent_state(parent, r["donor_unit"])
            receiver = parent_state(parent, r["receiver_unit"])
            path = baseline_path(out, r["unit"])
            if not path.exists():
                if not begin(path, 1):
                    return
                arrays, native, grad_score, audit = observe(
                    observer, r, m, receiver["h"], donor["h"], True
                )
                if not np.array_equal(arrays["q"], receiver["q"]):
                    raise ValueError("receiver query state changed")
                save_array_unit(
                    path,
                    {
                        "address": r,
                        "native": native,
                        "gradient_readout": grad_score,
                        "audit": audit,
                        "acquired_at": now(),
                        "source_sha256": barrier_sha,
                    },
                    arrays,
                )
                finish(1)
            meta = array_unit(path)
            if meta["address"] != r or meta["source_sha256"] != barrier_sha:
                raise ValueError("new reference address or checkpoint changed")
            for arm in spec["arms"]:
                physical = P.physical_patches(
                    donor["h"],
                    directions[r["query"]][arm],
                    spec["radius_relative_span_norm"],
                )
                for sign in ("plus", "minus"):
                    path = patch_path(out, r["unit"], arm, sign)
                    if path.exists():
                        continue
                    if not begin(path, 0):
                        return
                    native, audit = observe(
                        observer, r, m, receiver["h"], physical[sign], False
                    )
                    write_json(
                        path,
                        {
                            "address": {"unit": r["unit"], "arm": arm, "sign": sign},
                            "native": native,
                            "audit": audit,
                            "baseline_sha256": digest(baseline_path(out, r["unit"])),
                            "source_sha256": barrier_sha,
                            "acquired_at": now(),
                        },
                    )
                    finish(0)
            print(
                {
                    "unit": r["unit"],
                    "pair": r["pair"],
                    "query": r["query"],
                    "forwards": status["forward_calls"],
                    "backwards": status["backward_calls"],
                },
                flush=True,
            )
        write_json(
            out / "run_status.json",
            {**status, "status": "SOURCE_ACQUIRED" if source else "HELD_ACQUIRED"},
        )
    except Exception as exc:
        write_json(
            out / "run_status.json", {**status, "status": "ERROR", "error": repr(exc)}
        )
        raise


def freeze_source(out):
    """Checkpoint source acquisitions, not refit or invent a new source model."""
    m = check(out)
    if (out / "01/source_barrier.json").exists() or read_json(out / "run_status.json")[
        "status"
    ] != "SOURCE_ACQUIRED":
        raise ValueError("complete source acquisitions once, before held")
    hashes = {}
    for r in read_json(out / "00/design.json"):
        if r["diagonal"]:
            continue
        if r["role"] != "source":
            if baseline_path(out, r["unit"]).exists():
                raise ValueError("held data acquired before source checkpoint")
            continue
        p = baseline_path(out, r["unit"])
        array_unit(p)
        for f in (
            p,
            p.with_suffix(".npz"),
            *(
                patch_path(out, r["unit"], a, s)
                for a in m["spec"]["arms"]
                for s in ("plus", "minus")
            ),
        ):
            hashes[str(f.relative_to(out))] = digest(f)
    write_json(
        out / "01/source_barrier.json",
        {
            "status": "SOURCE_ACQUISITION_CHECKPOINT",
            "source_artifacts": hashes,
            "new_fits": 0,
            "parent_source_sha256": digest(
                ROOT / m["spec"]["parent"] / "01/source_manifest.json"
            ),
            "frozen_at": now(),
        },
    )
    write_json(
        out / "run_status.json",
        {**read_json(out / "run_status.json"), "status": "SOURCE_CHECKPOINTED"},
    )
    print({"status": "SOURCE_CHECKPOINTED", "new_fits": 0}, flush=True)


def evaluate(out):
    """Analyze every inherited/new corner and its frozen cold-field predictions."""
    m = check(out)
    spec = m["spec"]
    if read_json(out / "run_status.json")["status"] != "HELD_ACQUIRED":
        raise ValueError("complete actual acquisitions required")
    parent = ROOT / spec["parent"]
    fitted = fields(parent)
    design = read_json(out / "00/design.json")
    refs, actions, endpoints = {}, [], []
    with np.load(out / "00/directions.npz", allow_pickle=False) as bank:
        directions = {q: C.directions(bank[q + "_b"]) for q in spec["queries"]}
    for r in design:
        path = ref_path(out, r, m)
        base = array_unit(path)
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
            h, g = data["h"].copy(), data["gradient"].astype(float)
        donor = parent_state(parent, r["donor_unit"])["h"]
        if not np.array_equal(h, donor):
            raise ValueError("identical donor H not preserved")
        refs[r["unit"]] = (h, g, base["native"]["conditional_score"])
        for arm in spec["arms"]:
            physical = P.physical_patches(
                h, directions[r["query"]][arm], spec["radius_relative_span_norm"]
            )
            ends = {
                s: read_json(end_path(out, r, m, arm, s)) for s in ("plus", "minus")
            }
            moments = S.moment_map(4) @ physical["half_delta"]
            actions.append(
                {
                    k: r[k]
                    for k in (
                        "unit",
                        "context",
                        "pair",
                        "wording",
                        "query",
                        "role",
                        "donor_side",
                        "receiver_side",
                        "diagonal",
                    )
                }
                | {
                    "arm": arm,
                    **P.contrast(
                        base["native"]["conditional_score"],
                        ends["plus"]["native"]["conditional_score"],
                        ends["minus"]["native"]["conditional_score"],
                    ),
                    "physical_delta_norm": float(
                        np.linalg.norm(physical["half_delta"])
                    ),
                    "global_source": S.contract(
                        fitted[r["query"] + "_global"], moments
                    ),
                    "wording_source": S.contract(
                        fitted[r["query"] + f"_wording{r['wording']}"], moments
                    ),
                    "local_derivative": float(np.sum(g * physical["half_delta"])),
                }
            )
            for sign, end in ends.items():
                if any(end["audit"].values()) or end["baseline_sha256"] != digest(path):
                    raise ValueError("endpoint reference/complement audit failed")
                endpoints.append(
                    {
                        "unit": r["unit"],
                        "arm": arm,
                        "sign": sign,
                        "diagonal": r["diagonal"],
                        **end["native"],
                    }
                )
    metrics, decomposition, contrasts, arrays = [], [], [], {}
    for role in ("source", "validation", "application"):
        for q in spec["queries"]:
            for arm in spec["arms"]:
                for diagonal in (True, False):
                    items = [
                        a
                        for a in actions
                        if (a["role"], a["query"], a["arm"], a["diagonal"])
                        == (role, q, arm, diagonal)
                    ]
                    metrics.append(
                        {
                            "role": role,
                            "query": q,
                            "arm": arm,
                            "diagonal": diagonal,
                            **C.summarize(items),
                        }
                    )
    lookup = {(a["unit"], a["arm"]): a for a in actions}
    for pair in range(4):
        for w in range(4):
            for q in spec["queries"]:
                rows = [
                    r
                    for r in design
                    if (r["pair"], r["wording"], r["query"]) == (pair, w, q)
                ]
                by = {(r["donor_side"], r["receiver_side"]): r for r in rows}
                g = np.array(
                    [[refs[by[i, j]["unit"]][1] for j in range(2)] for i in range(2)]
                )
                for i in range(2):
                    if not np.array_equal(
                        refs[by[i, 0]["unit"]][0], refs[by[i, 1]["unit"]][0]
                    ):
                        raise ValueError("same-H fiber comparison failed")
                lam = np.einsum("ab,ijbd->ijad", np.linalg.inv(S.moment_map(4).T), g)
                task = np.stack(
                    [
                        np.einsum(
                            "ijad,ad->ij",
                            g,
                            directions[q][a] / np.linalg.norm(directions[q][a]),
                        )
                        for a in spec["arms"]
                    ],
                    axis=-1,
                )
                level = np.array(
                    [[refs[by[i, j]["unit"]][2] for j in range(2)] for i in range(2)]
                )
                address = {
                    "pair": pair,
                    "wording": w,
                    "query": q,
                    "role": rows[0]["role"],
                }
                items = [
                    ("gradient", "none", g),
                    ("covector", "none", lam),
                    ("task_gradient", "none", task),
                    ("native_level", "none", level),
                ]
                for arm in spec["arms"]:
                    a = [
                        [lookup[by[i, j]["unit"], arm] for j in range(2)]
                        for i in range(2)
                    ]
                    for units, method in (
                        ("raw_action", "odd_action"),
                        ("unit_action", "odd_action"),
                        ("local_unit_action", "local_derivative"),
                        ("cold_unit_action", "wording_source"),
                    ):
                        x = np.array(
                            [
                                [
                                    a[i][j][method]
                                    / (
                                        1
                                        if units == "raw_action"
                                        else a[i][j]["physical_delta_norm"]
                                    )
                                    for j in range(2)
                                ]
                                for i in range(2)
                            ]
                        )
                        items.append((units, arm, x))
                for space, arm, x in items:
                    key = f"p{pair}_w{w}_{q}_{space}_{arm}"
                    arrays[key] = x
                    parts, stats = F.split(x)
                    arrays.update({key + "_" + k: v for k, v in parts.items()})
                    decomposition.append(
                        {**address, "space": space, "arm": arm, **stats}
                    )
                    if space in (
                        "gradient",
                        "task_gradient",
                        "raw_action",
                        "unit_action",
                        "cold_unit_action",
                        "native_level",
                    ):
                        contrasts.extend(
                            {**address, "space": space, "arm": arm, **c}
                            for c in F.contrasts(x)
                        )
    for name, rows in (
        ("actions", actions),
        ("endpoints", endpoints),
        ("metrics", metrics),
        ("corner_decomposition", decomposition),
        ("fiber_contrasts", contrasts),
    ):
        write_csv(out / f"02/{name}.csv", rows)
    np.savez_compressed(out / "02/corner_arrays.npz", **arrays)
    write_json(
        out / "02/summary.json",
        {
            "status": "FINITE_HM31_ACQUIRED_NOT_GLOBAL",
            "counts": {
                "references": len(design),
                "new_references": 96,
                "actions": len(actions),
                "endpoints": len(endpoints),
                "four_corner_units": 48,
                "metrics": len(metrics),
                "decompositions": len(decomposition),
                "contrasts": len(contrasts),
                "new_fits": 0,
            },
            "calls": read_json(out / "run_status.json"),
            "scope": spec["scope"],
        },
    )
    paths = [
        p
        for p in out.rglob("*")
        if p.is_file()
        and not p.name.endswith(".tmp")
        and p.name not in ("run_status.json", "artifact_hashes.json")
    ]
    write_json(
        out / "02/artifact_hashes.json",
        {str(p.relative_to(out)): digest(p) for p in paths},
    )
    write_json(
        out / "run_status.json",
        {
            **read_json(out / "run_status.json"),
            "status": "FINITE_HM31_ACQUIRED_NOT_GLOBAL",
        },
    )
    print(read_json(out / "02/summary.json"), flush=True)


def main():
    """Run one guarded finite stage without restarting an existing call."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "source_calls",
            "freeze_source",
            "held_calls",
            "evaluate",
            "check",
        ),
    )
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    with collector_guard(args.out):
        if args.stage in ("source_calls", "held_calls"):
            calls(args.out, args.stage == "source_calls")
        elif args.stage == "check":
            print({"frozen_files": len(check(args.out)["files"])})
        else:
            globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
