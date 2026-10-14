#!/usr/bin/env python3
"""HM32 bounded H/R/Q cube with fixed weights and inherited natural-query corners."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_crossed_field as C
from grain_research.construction import math_position_support as P
from grain_research.construction import math_query_cube as U
from grain_research.construction import math_source_field as S
from grain_research.vector_mechanisms.direct_traits import native_readout
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_context_fiber import parent_state
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

SPEC = ROOT / "configs/grain_math_query_cube_v1.json"
PLAN = ROOT / "docs/GRAIN_QUERY_CONTEXT_CUBE_PLAN_V1.md"
OUT = ROOT / "results/grain_math_response_v1/query_context_cube_r1"
CODE = (
    "grain_research/construction/math_query_cube.py",
    "scripts/run_grain_math_query_cube.py",
    "scripts/verify_grain_math_query_cube.py",
    "tests/test_grain_math_query_cube.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)


def prepare(out):
    """Freeze384 corners/192 new Q swaps and every ancestry dependency."""
    if out.exists():
        raise FileExistsError("preserve existing packet and resume its stages")
    spec = read_json(SPEC)
    parent = ROOT / spec["parent"]
    pm = check(parent)
    verification = ROOT / spec["parent_verification"] / "summary.json"
    if read_json(verification)["status"] != "VERIFIED_FINITE_HM31_NOT_GLOBAL":
        raise ValueError("verified HM31 required")
    dependencies = dict(pm["files"])
    bound = ROOT / "results/grain_math_response_v1/context_fiber_descent_bound_r1"
    for name, sha in read_json(bound / "manifest.json")["files"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("frozen bound code changed " + name)
        dependencies[name] = sha
    for source, ledger_name in (
        (parent, "02/artifact_hashes.json"),
        (bound, "artifact_hashes.json"),
    ):
        for name, sha in read_json(source / ledger_name).items():
            if digest(source / name) != sha:
                raise ValueError("frozen parent changed " + name)
            dependencies[str((source / name).relative_to(ROOT))] = sha
        dependencies[str((source / ledger_name).relative_to(ROOT))] = digest(
            source / ledger_name
        )
    for path in (verification,):
        dependencies[str(path.relative_to(ROOT))] = digest(path)
    dependencies.update({name: digest(ROOT / name) for name in CODE})
    parent_rows = read_json(parent / "00/design.json")
    natural = ROOT / spec["natural_parent"]
    design = U.design(parent_rows, read_json(natural / "00/design.json"))
    if (
        len(design) != 384
        or sum(not r["inherited"] for r in design) != 192
        or list(C.ARMS) != spec["arms"]
        or len(fields(natural)) != 15
    ):
        raise ValueError("complete cube with frozen arms/fields required")
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
            "inputs": {
                "00/" + n: digest(out / "00" / n)
                for n in (
                    "design.json",
                    "directions.npz",
                    "queries.json",
                    "documents.json",
                )
            },
            "spec": spec,
            "alphabet": pm["alphabet"],
            "frozen_before_outputs": now(),
            "new_fits": 0,
            "index_group": "Z2^3 on design indices, not a group acting on natural hidden states",
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
    print({"status": "PREPARED", "references": 384, "new_references": 192}, flush=True)


def ref_path(out, row, m):
    """Resolve HM30/HM31 inherited corners without calling the decoder again."""
    if not row["inherited"]:
        return baseline_path(out, row["unit"])
    if row["diagonal"]:
        return baseline_path(ROOT / m["spec"]["natural_parent"], row["donor_unit"])
    return baseline_path(ROOT / m["spec"]["parent"], row["fiber_unit"])


def end_path(out, row, m, arm, sign):
    """Resolve each endpoint to its own original acquisition key."""
    if not row["inherited"]:
        return patch_path(out, row["unit"], arm, sign)
    if row["diagonal"]:
        return patch_path(
            ROOT / m["spec"]["natural_parent"], row["donor_unit"], arm, sign
        )
    return patch_path(ROOT / m["spec"]["parent"], row["fiber_unit"], arm, sign)


def observe(observer, row, m, receiver, donor_h, donor_q, gradient):
    """Replace H/Q only, audit R, and differentiate the postpatch leaf."""
    import torch

    slots = {}
    positions = row["spans"]["A_property"]["positions"]
    device = next(observer.model.parameters()).device

    def hook(_module, _args, output):
        changed, audit = U.replace_leaf(
            hidden_tensor(output),
            positions,
            receiver["h"],
            receiver["q"],
            donor_h,
            donor_q,
            gradient,
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
        arrays = {
            "h": slots["h"],
            "q": slots["q"],
            "receiver_h": receiver["h"],
            "receiver_q": receiver["q"],
            "gradient": g.detach().cpu().numpy().copy(),
        }
        return arrays, native, float(score.detach().cpu()), slots["audit"]
    finally:
        handle.remove()


def calls(out, source):
    """Acquire only opposite-Q corners, with one cumulative budget and journal."""
    m = check(out)
    spec = m["spec"]
    natural = ROOT / spec["natural_parent"]
    fields(natural)
    status = recover_in_flight(out, read_json(out / "run_status.json"))
    barrier_path = out / "01/source_barrier.json"
    if source == barrier_path.exists():
        raise ValueError("source before checkpoint, held after checkpoint")
    source_sha = None if source else digest(barrier_path)
    if not source:
        for name, sha in read_json(barrier_path)["source_artifacts"].items():
            if digest(out / name) != sha:
                raise ValueError("source checkpoint changed")
    design = read_json(out / "00/design.json")
    new = [r for r in design if not r["inherited"]]
    rows = [r for r in new if (r["role"] == "source") == source]
    n = sum(baseline_path(out, r["unit"]).exists() for r in new)
    e = sum(
        patch_path(out, r["unit"], a, s).exists()
        for r in new
        for a in spec["arms"]
        for s in ("plus", "minus")
    )
    if (status["forward_calls"], status["backward_calls"]) != (n + e, n):
        raise ValueError("committed artifacts and new-call journal disagree")
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
    elapsed, started = status["model_seconds"], time.monotonic()
    if elapsed >= spec["maximum_model_seconds"]:
        raise ValueError("cumulative model time exhausted")
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
            h = parent_state(natural, r["donor_unit"])["h"]
            q = parent_state(natural, r["query_donor_unit"])["q"]
            receiver = parent_state(natural, r["receiver_unit"])
            path = baseline_path(out, r["unit"])
            if not path.exists():
                if not begin(path, 1):
                    return
                arrays, native, grad_score, audit = observe(
                    observer, r, m, receiver, h, q, True
                )
                save_array_unit(
                    path,
                    {
                        "address": r,
                        "native": native,
                        "gradient_readout": grad_score,
                        "audit": audit,
                        "source_sha256": source_sha,
                        "acquired_at": now(),
                    },
                    arrays,
                )
                finish(1)
            meta = array_unit(path)
            if meta["address"] != r or meta["source_sha256"] != source_sha:
                raise ValueError("baseline address/checkpoint changed")
            for arm in spec["arms"]:
                physical = P.physical_patches(
                    h, directions[r["query"]][arm], spec["radius_relative_span_norm"]
                )
                for sign in ("plus", "minus"):
                    path = patch_path(out, r["unit"], arm, sign)
                    if path.exists():
                        continue
                    if not begin(path, 0):
                        return
                    native, audit = observe(
                        observer, r, m, receiver, physical[sign], q, False
                    )
                    write_json(
                        path,
                        {
                            "address": {"unit": r["unit"], "arm": arm, "sign": sign},
                            "native": native,
                            "audit": audit,
                            "baseline_sha256": digest(baseline_path(out, r["unit"])),
                            "source_sha256": source_sha,
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
    """Checkpoint all source calls with zero new fits before held acquisition."""
    m = check(out)
    if (out / "01/source_barrier.json").exists() or read_json(out / "run_status.json")[
        "status"
    ] != "SOURCE_ACQUIRED":
        raise ValueError("complete sources before one checkpoint")
    hashes = {}
    for r in read_json(out / "00/design.json"):
        if r["inherited"]:
            continue
        if r["role"] != "source":
            if baseline_path(out, r["unit"]).exists():
                raise ValueError("held acquisition before source checkpoint")
            continue
        p = baseline_path(out, r["unit"])
        array_unit(p)
        for path in (
            p,
            p.with_suffix(".npz"),
            *(
                patch_path(out, r["unit"], a, s)
                for a in m["spec"]["arms"]
                for s in ("plus", "minus")
            ),
        ):
            hashes[str(path.relative_to(out))] = digest(path)
    write_json(
        out / "01/source_barrier.json",
        {
            "status": "SOURCE_ACQUISITION_CHECKPOINT",
            "source_artifacts": hashes,
            "new_fits": 0,
            "parent_source_sha256": digest(
                ROOT / m["spec"]["natural_parent"] / "01/source_manifest.json"
            ),
            "frozen_at": now(),
        },
    )
    write_json(
        out / "run_status.json",
        {**read_json(out / "run_status.json"), "status": "SOURCE_CHECKPOINTED"},
    )
    print({"status": "SOURCE_CHECKPOINTED", "new_fits": 0}, flush=True)


def pool(rows):
    """Average finite-cell energies before taking ratios, not average percentages."""
    keys = [k for k in rows[0] if k.endswith(("_energy", "_mse_bound"))]
    result = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    total, all_energy = result["total_centered_energy"], result["mean_square_energy"]
    for label in U.LABELS[1:]:
        result[label + "_share"] = result[label + "_energy"] / total if total else None
    for label in ("H", "HQ", "HR"):
        result[label + "_only_relative_rmse_bound"] = (
            float(np.sqrt(result[label + "_only_mse_bound"] / all_energy))
            if all_energy
            else None
        )
    return {"cubes": len(rows), **result}


def evaluate(out):
    """Compile no new model; decompose all actual cube actions and bounds."""
    m = check(out)
    spec = m["spec"]
    if read_json(out / "run_status.json")["status"] != "HELD_ACQUIRED":
        raise ValueError("complete actual acquisition required")
    design = read_json(out / "00/design.json")
    natural = ROOT / spec["natural_parent"]
    fitted = fields(natural)
    with np.load(out / "00/directions.npz", allow_pickle=False) as bank:
        directions = {q: C.directions(bank[q + "_b"]) for q in spec["queries"]}
    refs, actions, endpoints = {}, [], []
    for r in design:
        p = ref_path(out, r, m)
        base = array_unit(p)
        with np.load(p.with_suffix(".npz"), allow_pickle=False) as data:
            h, q, g = data["h"].copy(), data["q"].copy(), data["gradient"].astype(float)
        if not np.array_equal(
            h, parent_state(natural, r["donor_unit"])["h"]
        ) or not np.array_equal(q, parent_state(natural, r["query_donor_unit"])["q"]):
            raise ValueError("exact H/Q donors not preserved")
        refs[r["unit"]] = (h, q, g, base["native"]["conditional_score"])
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
                        "query_side",
                        "inherited",
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
                if any(end["audit"].values()) or end["baseline_sha256"] != digest(p):
                    raise ValueError("endpoint reference or H/Q/R audit failed")
                endpoints.append(
                    {
                        "unit": r["unit"],
                        "arm": arm,
                        "sign": sign,
                        "inherited": r["inherited"],
                        **end["native"],
                    }
                )
    metrics, decomposition, contrasts, arrays = [], [], [], {}
    for role in ("source", "validation", "application"):
        for q in spec["queries"]:
            for arm in spec["arms"]:
                for inherited in (True, False):
                    items = [
                        a
                        for a in actions
                        if (a["role"], a["query"], a["arm"], a["inherited"])
                        == (role, q, arm, inherited)
                    ]
                    metrics.append(
                        {
                            "role": role,
                            "query": q,
                            "arm": arm,
                            "inherited": inherited,
                            **C.summarize(items),
                        }
                    )
    lookup = {(a["unit"], a["arm"]): a for a in actions}
    for pair in range(4):
        for w in range(4):
            for q in spec["queries"]:
                selected = [
                    r
                    for r in design
                    if (r["pair"], r["wording"], r["query"]) == (pair, w, q)
                ]
                by = {
                    (r["donor_side"], r["receiver_side"], r["query_side"]): r
                    for r in selected
                }
                if len(by) != 8:
                    raise ValueError("incomplete cube")
                g = np.array([refs[by[s]["unit"]][2] for s in U.BITS]).reshape(
                    2, 2, 2, 4, -1
                )
                lam = np.einsum("ab,ijkbd->ijkad", np.linalg.inv(S.moment_map(4).T), g)
                task = np.stack(
                    [
                        np.einsum(
                            "ijkad,ad->ijk",
                            g,
                            directions[q][a] / np.linalg.norm(directions[q][a]),
                        )
                        for a in spec["arms"]
                    ],
                    axis=-1,
                )
                level = np.array([refs[by[s]["unit"]][3] for s in U.BITS]).reshape(
                    2, 2, 2
                )
                address = {
                    "pair": pair,
                    "wording": w,
                    "query": q,
                    "role": selected[0]["role"],
                }
                items = [
                    ("gradient", "none", g),
                    ("covector", "none", lam),
                    ("task_gradient", "none", task),
                    ("native_level", "none", level),
                ]
                for arm in spec["arms"]:
                    rows = [lookup[by[s]["unit"], arm] for s in U.BITS]
                    for space, method in (
                        ("raw_action", "odd_action"),
                        ("unit_action", "odd_action"),
                        ("local_unit_action", "local_derivative"),
                        ("cold_unit_action", "wording_source"),
                    ):
                        x = np.array(
                            [
                                r[method]
                                / (
                                    1
                                    if space == "raw_action"
                                    else r["physical_delta_norm"]
                                )
                                for r in rows
                            ]
                        ).reshape(2, 2, 2)
                        items.append((space, arm, x))
                for space, arm, x in items:
                    key = f"p{pair}_w{w}_{q}_{space}_{arm}"
                    arrays[key] = x
                    coefficients, stats = U.split(x)
                    arrays[key + "_coefficients"] = coefficients
                    decomposition.append(
                        {**address, "space": space, "arm": arm, **stats}
                    )
                    if space in (
                        "gradient",
                        "task_gradient",
                        "native_level",
                        "raw_action",
                        "unit_action",
                        "cold_unit_action",
                    ):
                        contrasts.extend(
                            {**address, "space": space, "arm": arm, **c}
                            for c in U.contrasts(x)
                        )
    pooled = []
    for role in ("source", "validation", "application"):
        for q in spec["queries"]:
            for space, arm in dict.fromkeys(
                (r["space"], r["arm"]) for r in decomposition
            ):
                rows = [
                    r
                    for r in decomposition
                    if (r["role"], r["query"], r["space"], r["arm"])
                    == (role, q, space, arm)
                ]
                pooled.append(
                    {"role": role, "query": q, "space": space, "arm": arm, **pool(rows)}
                )
    for name, rows in (
        ("actions", actions),
        ("endpoints", endpoints),
        ("metrics", metrics),
        ("cube_decomposition", decomposition),
        ("cube_contrasts", contrasts),
        ("pooled_decomposition", pooled),
    ):
        write_csv(out / f"02/{name}.csv", rows)
    np.savez_compressed(out / "02/cube_arrays.npz", **arrays)
    write_json(
        out / "02/summary.json",
        {
            "status": "FINITE_HM32_ACQUIRED_NOT_GLOBAL",
            "counts": {
                "references": len(design),
                "new_references": 192,
                "actions": len(actions),
                "endpoints": len(endpoints),
                "cubes": 48,
                "metrics": len(metrics),
                "decompositions": len(decomposition),
                "contrasts": len(contrasts),
                "pooled_rows": len(pooled),
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
        and not p.name.startswith("._")
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
            "status": "FINITE_HM32_ACQUIRED_NOT_GLOBAL",
        },
    )
    print(read_json(out / "02/summary.json"), flush=True)


def main():
    """Dispatch one guarded finite stage, never automatically replay or refit."""
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
