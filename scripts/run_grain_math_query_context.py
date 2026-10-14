#!/usr/bin/env python3
"""HM33 exact H/B/P/Q carrier localization, fixed weights and no new fits."""

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
from grain_research.construction import math_query_context as U
from grain_research.construction import math_source_field as S
from grain_research.vector_mechanisms.direct_traits import native_readout
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts import run_grain_math_query_cube as OLD
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

SPEC = ROOT / "configs/grain_math_query_context_v1.json"
PLAN = ROOT / "docs/GRAIN_EARLY_QUERY_CONTEXT_LOCALIZATION_PLAN_V1.md"
OUT = ROOT / "results/grain_math_response_v1/early_query_context_localization_r1"
CODE = (
    "grain_research/construction/math_query_context.py",
    "scripts/run_grain_math_query_context.py",
    "scripts/verify_grain_math_query_context.py",
    "tests/test_grain_math_query_context.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)


def prepare(out):
    """Freeze768 corners and all inherited ledgers before any new capture."""
    if out.exists():
        raise FileExistsError("preserve and resume existing packet")
    spec = read_json(SPEC)
    old = ROOT / spec["parent"]
    pm = check(old)
    verification = ROOT / spec["parent_verification"]
    if (
        read_json(verification / "summary.json")["status"]
        != "VERIFIED_FINITE_HM32_NOT_GLOBAL"
    ):
        raise ValueError("verified predecessor required")
    dependencies = dict(pm["files"])
    for manifest in (
        verification / "csv_adapter_manifest.json",
        ROOT / spec["parent_alias"] / "manifest.json",
    ):
        for name, sha in read_json(manifest)["files"].items():
            if digest(ROOT / name) != sha:
                raise ValueError("frozen ancestor code changed " + name)
            dependencies[name] = sha
    for source, ledger_name in (
        (old, "02/artifact_hashes.json"),
        (verification, "artifact_hashes.json"),
        (ROOT / spec["parent_alias"], "artifact_hashes.json"),
    ):
        for name, sha in read_json(source / ledger_name).items():
            path = source / name
            if digest(path) != sha:
                raise ValueError("frozen ancestor changed " + str(path))
            dependencies[str(path.relative_to(ROOT))] = sha
        dependencies[str((source / ledger_name).relative_to(ROOT))] = digest(
            source / ledger_name
        )
    dependencies.update({name: digest(ROOT / name) for name in CODE})
    natural = ROOT / spec["natural_parent"]
    rows = U.design(
        read_json(old / "00/design.json"), read_json(natural / "00/design.json")
    )
    if (
        len(rows) != 768
        or sum(not r["inherited"] for r in rows) != 384
        or len(fields(natural)) != 15
        or list(C.ARMS) != spec["arms"]
    ):
        raise ValueError("complete frozen design/fields required")
    for name in ("00/code", "01/captures", "01/baselines", "01/patches", "02"):
        (out / name).mkdir(parents=True, exist_ok=True)
    write_json(out / "00/design.json", rows)
    for name in ("directions.npz", "queries.json", "documents.json"):
        (out / "00" / name).write_bytes((old / "00" / name).read_bytes())
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
            "index_group": "Z2^4 on experimental indices, not natural vector group action",
        },
    )
    write_json(
        out / "run_status.json",
        {
            "status": "PREPARED",
            "forward_calls": 0,
            "backward_calls": 0,
            "model_seconds": 0.0,
            "in_flight": None,
        },
    )
    print(
        {
            "status": "PREPARED",
            "references": 768,
            "new_references": 384,
            "natural_captures": 96,
        },
        flush=True,
    )


def capture_path(out, unit):
    """Address one replay of an original natural input, not a new material."""
    return out / "01/captures" / f"unit_{unit:03d}.json"


def state(out, unit):
    """Read hash-verified natural H/P/Q arrays captured by this packet."""
    path = capture_path(out, unit)
    array_unit(path)
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
        return {k: arrays[k].copy() for k in ("h", "p", "q")}


def paths(out, row, m):
    """Keep exact original HM32 ownership for inherited P=B corners."""
    if not row["inherited"]:
        return baseline_path(out, row["unit"]), lambda a, s: patch_path(
            out, row["unit"], a, s
        )
    old = ROOT / m["spec"]["parent"]
    original = read_json(old / "00/design.json")[row["cube_unit"]]
    pm = read_json(old / "00/manifest.json")
    return OLD.ref_path(old, original, pm), lambda a, s: OLD.end_path(
        old, original, pm, a, s
    )


def capture(observer, row, m):
    """Observe missing P states without changing the natural forward computation."""
    slots = {}
    hp, pp = row["spans"]["A_property"]["positions"], U.positions(row)

    def hook(_module, _args, output):
        hidden = hidden_tensor(output)
        slots.update(
            {
                k: hidden[0, address].detach().cpu().numpy().copy()
                for k, address in (("h", hp), ("p", pp), ("q", -1))
            }
        )

    handle = observer.model.model.layers[m["spec"]["layer"]].register_forward_hook(hook)
    try:
        native = forward(observer, row["input_ids"], m["alphabet"])
    finally:
        handle.remove()
    original = parent_state(ROOT / m["spec"]["natural_parent"], row["unit"])
    if any(not np.array_equal(slots[k], original[k]) for k in ("h", "q")):
        raise ValueError("natural H/Q replay changed")
    original_native = read_json(
        baseline_path(ROOT / m["spec"]["natural_parent"], row["unit"])
    )["native"]
    if native != original_native:
        raise ValueError("natural decoder readout replay changed")
    return slots, native


def observe(observer, row, m, receiver, donors, gradient):
    """Patch H/P/Q, leave B exact, and differentiate only the postpatch leaf."""
    import torch

    slots = {}
    hp, pp = row["spans"]["A_property"]["positions"], row["early_query_positions"]
    device = next(observer.model.parameters()).device

    def hook(_module, _args, output):
        changed, audit = U.replace_leaf(
            hidden_tensor(output), hp, pp, receiver, donors, gradient
        )
        slots.update(
            {
                "leaf": changed,
                "audit": audit,
                **{
                    k: changed[0, address].detach().cpu().numpy().copy()
                    for k, address in (("h", hp), ("p", pp), ("q", -1))
                },
            }
        )
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    handle = observer.model.model.layers[m["spec"]["layer"]].register_forward_hook(hook)
    try:
        if not gradient:
            return forward(observer, row["input_ids"], m["alphabet"]), slots["audit"]
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
        g = torch.autograd.grad(score, slots["leaf"])[0][0, hp]
        arrays = {k: slots[k] for k in ("h", "p", "q")}
        arrays.update({"receiver_" + k: receiver[k] for k in ("h", "p", "q")})
        arrays["gradient"] = g.detach().cpu().numpy().copy()
        return (
            arrays,
            native_readout(logits.detach().float().cpu().numpy(), m["alphabet"]),
            float(score.detach().cpu()),
            slots["audit"],
        )
    finally:
        handle.remove()


def calls(out, source):
    """Acquire one guarded stage, including P captures and exact call accounting."""
    m = check(out)
    spec = m["spec"]
    status = recover_in_flight(out, read_json(out / "run_status.json"))
    barrier = out / "01/source_barrier.json"
    if source == barrier.exists():
        raise ValueError("source before checkpoint, held after checkpoint")
    source_sha = None if source else digest(barrier)
    if not source:
        for name, sha in read_json(barrier)["source_artifacts"].items():
            if digest(out / name) != sha:
                raise ValueError("source checkpoint changed")
    naturals = read_json(ROOT / spec["natural_parent"] / "00/design.json")
    new = [r for r in read_json(out / "00/design.json") if not r["inherited"]]
    rows = [r for r in new if (r["role"] == "source") == source]
    captures = [r for r in naturals if (r["role"] == "source") == source]
    n = sum(baseline_path(out, r["unit"]).exists() for r in new)
    e = sum(
        patch_path(out, r["unit"], a, s).exists()
        for r in new
        for a in spec["arms"]
        for s in ("plus", "minus")
    )
    c = sum(capture_path(out, r["unit"]).exists() for r in naturals)
    if (status["forward_calls"], status["backward_calls"]) != (n + e + c, n):
        raise ValueError("journal and committed artifacts disagree")
    elapsed, started = status["model_seconds"], time.monotonic()
    observer = load_observer()
    if any(p.requires_grad for p in observer.model.parameters()):
        raise ValueError("fixed weights required")
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
        for r in captures:
            path = capture_path(out, r["unit"])
            if path.exists():
                array_unit(path)
                continue
            if not begin(path, 0):
                return
            arrays, native = capture(observer, r, m)
            save_array_unit(
                path,
                {
                    "address": r,
                    "early_query_positions": U.positions(r),
                    "early_query_token_ids": [
                        r["input_ids"][i] for i in U.positions(r)
                    ],
                    "native": native,
                    "source_sha256": source_sha,
                    "acquired_at": now(),
                },
                arrays,
            )
            finish(0)
        for r in rows:
            receiver = state(out, r["receiver_unit"])
            donors = {
                "h": state(out, r["donor_unit"])["h"],
                "p": state(out, r["early_query_donor_unit"])["p"],
                "q": state(out, r["query_donor_unit"])["q"],
            }
            path = baseline_path(out, r["unit"])
            if not path.exists():
                if not begin(path, 1):
                    return
                arrays, native, grad_score, audit = observe(
                    observer, r, m, receiver, donors, True
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
            if array_unit(path)["address"] != r:
                raise ValueError("reference address changed")
            for arm in spec["arms"]:
                physical = P.physical_patches(
                    donors["h"],
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
                        observer, r, m, receiver, {**donors, "h": physical[sign]}, False
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
    """Seal source captures and interventions before obtaining any held P state."""
    m = check(out)
    barrier = out / "01/source_barrier.json"
    status = read_json(out / "run_status.json")
    if (
        barrier.exists()
        or status["status"] != "SOURCE_ACQUIRED"
        or (status["forward_calls"], status["backward_calls"]) != (1008, 192)
    ):
        raise ValueError("complete source stage and unique checkpoint required")
    hashes = {}
    for r in read_json(ROOT / m["spec"]["natural_parent"] / "00/design.json"):
        path = capture_path(out, r["unit"])
        if r["role"] == "source":
            array_unit(path)
            for p in (path, path.with_suffix(".npz")):
                hashes[str(p.relative_to(out))] = digest(p)
        elif path.exists():
            raise ValueError("held capture before checkpoint")
    for r in read_json(out / "00/design.json"):
        if r["inherited"]:
            continue
        path = baseline_path(out, r["unit"])
        if r["role"] != "source":
            if path.exists():
                raise ValueError("held reference before checkpoint")
            continue
        array_unit(path)
        for p in (
            path,
            path.with_suffix(".npz"),
            *(
                patch_path(out, r["unit"], a, s)
                for a in m["spec"]["arms"]
                for s in ("plus", "minus")
            ),
        ):
            hashes[str(p.relative_to(out))] = digest(p)
    write_json(
        barrier,
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
    write_json(out / "run_status.json", {**status, "status": "SOURCE_CHECKPOINTED"})
    print(
        {"status": "SOURCE_CHECKPOINTED", "source_hashes": len(hashes), "new_fits": 0},
        flush=True,
    )


def pool(rows):
    """Pool energies first; ratios concern this common finite product design."""
    result = {
        k: float(np.mean([r[k] for r in rows]))
        for k in rows[0]
        if k.endswith(("_energy", "_mse_bound"))
    }
    total, all_energy = result["total_centered_energy"], result["mean_square_energy"]
    for label in U.LABELS[1:]:
        result[label + "_share"] = result[label + "_energy"] / total if total else None
    for label in ("HQ", "HPQ", "HBQ"):
        result[label + "_only_relative_rmse_bound"] = (
            float(np.sqrt(result[label + "_only_mse_bound"] / all_energy))
            if all_energy
            else None
        )
    return {"cubes": len(rows), **result}


def evaluate(out):
    """Evaluate all corners without fitting or making additional model calls."""
    m = check(out)
    spec = m["spec"]
    if read_json(out / "run_status.json")["status"] != "HELD_ACQUIRED":
        raise ValueError("complete actual acquisition required")
    design = read_json(out / "00/design.json")
    fitted = fields(ROOT / spec["natural_parent"])
    with np.load(out / "00/directions.npz", allow_pickle=False) as bank:
        directions = {q: C.directions(bank[q + "_b"]) for q in spec["queries"]}
    refs, actions, endpoints = {}, [], []
    for r in design:
        path, endpoint = paths(out, r, m)
        base = array_unit(path)
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
            h, q, g = data["h"].copy(), data["q"].copy(), data["gradient"].astype(float)
            if not r["inherited"] and not np.array_equal(
                data["p"], state(out, r["early_query_donor_unit"])["p"]
            ):
                raise ValueError("P donor changed")
        if not np.array_equal(
            h, state(out, r["donor_unit"])["h"]
        ) or not np.array_equal(q, state(out, r["query_donor_unit"])["q"]):
            raise ValueError("H/Q donor changed")
        refs[r["unit"]] = (h, q, g, base["native"]["conditional_score"])
        for arm in spec["arms"]:
            physical = P.physical_patches(
                h, directions[r["query"]][arm], spec["radius_relative_span_norm"]
            )
            ends = {s: read_json(endpoint(arm, s)) for s in ("plus", "minus")}
            moments = S.moment_map(4) @ physical["half_delta"]
            actions.append(
                {
                    **{
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
                            "early_query_side",
                            "query_side",
                            "inherited",
                        )
                    },
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
                    raise ValueError("endpoint mapping or H/P/Q/B preservation failed")
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
                    (
                        r["donor_side"],
                        r["receiver_side"],
                        r["early_query_side"],
                        r["query_side"],
                    ): r
                    for r in selected
                }
                if len(by) != 16:
                    raise ValueError("complete16 corners required")
                g = np.array([refs[by[s]["unit"]][2] for s in U.BITS]).reshape(
                    2, 2, 2, 2, 4, -1
                )
                lam = np.einsum(
                    "ab,ijklbd->ijklad", np.linalg.inv(S.moment_map(4).T), g
                )
                task = np.stack(
                    [
                        np.einsum(
                            "ijklad,ad->ijkl",
                            g,
                            directions[q][a] / np.linalg.norm(directions[q][a]),
                        )
                        for a in spec["arms"]
                    ],
                    axis=-1,
                )
                level = np.array([refs[by[s]["unit"]][3] for s in U.BITS]).reshape(
                    2, 2, 2, 2
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
                        ).reshape(2, 2, 2, 2)
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
            "status": "FINITE_HM33_ACQUIRED_NOT_GLOBAL",
            "counts": {
                "natural_captures": 96,
                "references": len(design),
                "new_references": 384,
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
    artifacts = [
        p
        for p in out.rglob("*")
        if p.is_file()
        and not p.name.startswith("._")
        and not p.name.endswith(".tmp")
        and p.name not in ("run_status.json", "artifact_hashes.json")
    ]
    write_json(
        out / "02/artifact_hashes.json",
        {str(p.relative_to(out)): digest(p) for p in artifacts},
    )
    write_json(
        out / "run_status.json",
        {
            **read_json(out / "run_status.json"),
            "status": "FINITE_HM33_ACQUIRED_NOT_GLOBAL",
        },
    )
    print(read_json(out / "02/summary.json"), flush=True)


def main():
    """One explicit guarded stage per invocation, no automatic replay/refit."""
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
