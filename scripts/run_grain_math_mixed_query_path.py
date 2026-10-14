#!/usr/bin/env python3
"""HM35 bounded actual mixed derivatives along frozen early-query paths."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_crossed_field as C
from grain_research.construction import math_mixed_query_path as M
from grain_research.construction import math_position_support as P
from grain_research.construction import math_query_context as U
from grain_research.vector_mechanisms.direct_traits import native_readout
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts import run_grain_math_query_context as OLD
from scripts.run_grain_math_position_support import array_unit, check, now
from scripts.run_grain_math_query_response import load_model
from scripts.run_grain_math_response import write_csv, write_json
from scripts.run_grain_math_span_binding import (
    hidden_tensor,
    load_observer,
    save_array_unit,
)
from scripts.run_grain_upstream_routes import collector_guard
from scripts.verify_grain_math_crossed_field import table

SPEC = ROOT / "configs/grain_math_mixed_query_path_v1.json"
PLAN = ROOT / "docs/GRAIN_MIXED_QUERY_PATH_PLAN_V1.md"
OUT = ROOT / "results/grain_math_response_v1/mixed_query_path_r1"
CODE = (
    "grain_research/construction/math_mixed_query_path.py",
    "scripts/run_grain_math_mixed_query_path.py",
    "scripts/verify_grain_math_mixed_query_path.py",
    "tests/test_grain_math_mixed_query_path.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)


def node_path(out, path, node):
    """Address one actual node with both semantic path and fixed quadrature index."""
    return out / "01/nodes" / f"path_{path:03d}_node_{node}.json"


def prepare(out):
    """Freeze code, inputs, source projector and all design addresses before calls."""
    if out.exists():
        raise FileExistsError("preserve existing packet; resume explicitly")
    spec = read_json(SPEC)
    parent, operator = ROOT / spec["parent"], ROOT / spec["operator_parent"]
    pm, om = check(parent), check(operator)
    verified = ROOT / spec["operator_verification"] / "summary.json"
    if not read_json(verified)["status"].startswith("VERIFIED_FINITE_HM34"):
        raise ValueError("verified frozen HM34 projector required")
    files = {**pm["files"], **om["files"]}
    for base in (parent, operator):
        for name, sha in read_json(base / "02/artifact_hashes.json").items():
            p = base / name
            if digest(p) != sha:
                raise ValueError("inherited result changed " + str(p))
            files[str(p.relative_to(ROOT))] = sha
    for name, sha in read_json(operator / "01/source_barrier.json")["files"].items():
        p = operator / name
        if digest(p) != sha:
            raise ValueError("source model changed")
        files[str(p.relative_to(ROOT))] = sha
    files[str(verified.relative_to(ROOT))] = digest(verified)
    rows = M.design(read_json(parent / "00/design.json"), spec["queries"])
    if (
        len(rows) != spec["paths"]
        or sum(r["role"] == "source" for r in rows) != spec["source_paths"]
    ):
        raise ValueError("predeclared complete design required")
    if list(M.NODES) != spec["nodes"] or spec["arms"] != list(C.ARMS):
        raise ValueError("fixed nodes and original arms required")
    for name in ("00/code", "01/nodes", "02"):
        (out / name).mkdir(parents=True, exist_ok=True)
    write_json(out / "00/design.json", rows)
    for name in CODE:
        files[name] = digest(ROOT / name)
        dst = out / "00/code" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / name).read_bytes())
    write_json(
        out / "00/manifest.json",
        {
            "spec": spec,
            "alphabet": pm["alphabet"],
            "files": files,
            "inputs": {"00/design.json": digest(out / "00/design.json")},
            "frozen_at": now(),
            "new_fits": 0,
        },
    )
    write_json(out / "run_status.json", M.new_status())
    print(
        {"status": "PREPARED", "paths": len(rows), "nodes": len(rows) * 3}, flush=True
    )


def inputs(out, path, m):
    """Build exact physical directions from unchanged source geometry and donors."""
    spec, row = m["spec"], path["left"]
    parent = ROOT / spec["parent"]
    receiver = OLD.state(parent, row["receiver_unit"])
    h = OLD.state(parent, row["donor_unit"])["h"]
    q = OLD.state(parent, row["query_donor_unit"])["q"]
    p0 = OLD.state(parent, row["early_query_donor_unit"])["p"]
    p1 = OLD.state(parent, path["right"]["early_query_donor_unit"])["p"]
    delta = p0.astype(float) - p1.astype(float)
    model = load_model(ROOT / spec["operator_parent"], row["query"])
    supported, unsupported = M.project(model, delta, row["early_query_token_ids"])
    with np.load(parent / "00/directions.npz", allow_pickle=False) as bank:
        directions = C.directions(bank[row["query"] + "_b"])
    dh = np.stack(
        [
            P.physical_patches(h, directions[a], spec["radius_relative_span_norm"])[
                "half_delta"
            ]
            for a in spec["arms"]
        ]
    )
    return (
        receiver,
        {"h": h, "q": q},
        {
            "p0": p0,
            "p1": p1,
            "delta_p": delta,
            "supported": supported,
            "unsupported": unsupported,
            "delta_h": dh,
        },
    )


def observe(observer, row, m, receiver, donors, delta_h, journal):
    """One real forward and three actual gradient calls on the same leaf graph."""
    import torch

    hp, pp = row["spans"]["A_property"]["positions"], row["early_query_positions"]
    device = next(observer.model.parameters()).device
    slots = {}

    def hook(_module, _args, output):
        leaf, audit = U.replace_leaf(
            hidden_tensor(output), hp, pp, receiver, donors, True
        )
        slots.update(leaf=leaf, audit=audit)
        return (leaf, *output[1:]) if isinstance(output, tuple) else leaf

    handle = observer.model.model.layers[m["spec"]["layer"]].register_forward_hook(hook)
    try:
        ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
        journal("forward", "attempted")
        output = observer.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )
        observer.calls += 1
        torch.mps.synchronize()
        journal("forward", "completed")
        logits = output.logits[0, -1]
        values = torch.tensor(
            m["alphabet"]["values"], dtype=torch.float32, device=device
        )
        score = (
            torch.softmax(logits[m["alphabet"]["token_ids"]].float(), dim=0) * values
        ).sum()
        journal("first", "attempted")
        gradient = torch.autograd.grad(score, slots["leaf"], create_graph=True)[0][
            0, hp
        ]
        torch.mps.synchronize()
        journal("first", "completed")
        covectors = []
        for i, dh in enumerate(delta_h):
            scalar = (
                gradient * torch.tensor(dh, dtype=gradient.dtype, device=device)
            ).sum()
            journal("second", "attempted")
            second = torch.autograd.grad(scalar, slots["leaf"], retain_graph=i == 0)[0][
                0, pp
            ]
            torch.mps.synchronize()
            journal("second", "completed")
            covectors.append(second.detach().cpu().numpy().copy())
        arrays = {
            **donors,
            "gradient": gradient.detach().cpu().numpy().copy(),
            "p_covectors": np.stack(covectors),
        }
        return (
            arrays,
            native_readout(logits.detach().float().cpu().numpy(), m["alphabet"]),
            float(score.detach().cpu()),
            slots["audit"],
        )
    finally:
        handle.remove()


def endpoint_replay(parent, row, m, arrays, native, score):
    """Validate exact cached endpoint scores and first H gradients, without refit."""
    reference, _ = OLD.paths(parent, row, read_json(parent / "00/manifest.json"))
    meta = array_unit(reference)
    with np.load(reference.with_suffix(".npz"), allow_pickle=False) as a:
        error = float(
            np.max(abs(arrays["gradient"].astype(float) - a["gradient"].astype(float)))
        )
        if not np.array_equal(arrays["h"], a["h"]):
            raise ValueError("endpoint H donor replay failed")
    score_error = abs(score - meta["gradient_readout"])
    if (
        native != meta["native"]
        or error > m["spec"]["endpoint_gradient_absolute_tolerance"]
        or score_error > m["spec"]["endpoint_score_absolute_tolerance"]
    ):
        raise ValueError(
            f"endpoint replay failed: gradient={error}, score={score_error}"
        )
    return {
        "gradient_max_error": error,
        "score_error": score_error,
        "reference": str(reference.relative_to(ROOT)),
    }


def calls(out, source):
    """Acquire a finite stage; partial/uncertain derivative calls never auto-replay."""
    m = check(out)
    spec, status = m["spec"], read_json(out / "run_status.json")
    expected = "PREPARED" if source else "SOURCE_CHECKPOINTED"
    if status["status"] != expected or status["in_flight"] is not None:
        raise ValueError(
            "untouched finite stage required; uncertain calls need a separate decision"
        )
    barrier = out / "01/source_barrier.json"
    if barrier.exists() == source:
        raise ValueError("source then unique checkpoint then application required")
    source_sha = None if source else digest(barrier)
    if not source:
        for name, sha in read_json(barrier)["files"].items():
            if digest(out / name) != sha:
                raise ValueError("source checkpoint changed")
    start, elapsed = time.monotonic(), status["model_seconds"]
    observer = load_observer()
    if (
        str(next(observer.model.parameters()).device) != spec["device"]
        or observer.model.config._attn_implementation != spec["attention_backend"]
        or any(p.requires_grad for p in observer.model.parameters())
    ):
        raise ValueError("frozen eager MPS observer required")
    write_json(
        out / ("01/source_observer.json" if source else "01/application_observer.json"),
        observer.manifest(),
    )

    def persist():
        status["model_seconds"] = elapsed + time.monotonic() - start
        write_json(out / "run_status.json", status)

    def journal(kind, phase):
        nonlocal status
        if phase == "attempted":
            if elapsed + time.monotonic() - start >= spec["maximum_model_seconds"]:
                raise TimeoutError("fixed cumulative model-compute boundary")
            status = M.reserve(status, kind, spec[f"maximum_{kind}_calls"])
        else:
            status[f"{kind}_completed"] += 1
        persist()

    try:
        paths = [
            p
            for p in read_json(out / "00/design.json")
            if (p["role"] == "source") == source
        ]
        for path in paths:
            receiver, donors, fixed = inputs(out, path, m)
            for index, t in enumerate(M.NODES):
                dst = node_path(out, path["path"], index)
                if dst.exists() or dst.with_suffix(".npz").exists():
                    raise ValueError("unexpected prior node; no silent replay")
                status.update(
                    status="SOURCE_CALLS" if source else "APPLICATION_CALLS",
                    in_flight={
                        "path": path["path"],
                        "node": index,
                        "started_at": now(),
                    },
                )
                persist()
                node_donors = {
                    **donors,
                    "p": M.interpolate(fixed["p0"], fixed["p1"], t),
                }
                arrays, native, score, audit = observe(
                    observer,
                    path["left"],
                    m,
                    receiver,
                    node_donors,
                    fixed["delta_h"],
                    journal,
                )
                replay = None
                if t in (0, 1):
                    row = path["right"] if t == 0 else path["left"]
                    replay = endpoint_replay(
                        ROOT / spec["parent"], row, m, arrays, native, score
                    )
                arrays.update(fixed)
                arrays["mixed_actions"] = M.mixed_actions(
                    arrays["p_covectors"],
                    fixed["delta_p"],
                    fixed["supported"],
                    fixed["unsupported"],
                )
                save_array_unit(
                    dst,
                    {
                        "address": path,
                        "node": index,
                        "t": t,
                        "native": native,
                        "gradient_readout": score,
                        "audit": audit,
                        "endpoint_replay": replay,
                        "source_sha256": source_sha,
                        "acquired_at": now(),
                    },
                    arrays,
                )
                status["committed_nodes"] += 1
                status["in_flight"] = None
                persist()
            print(
                {
                    "path": path["path"],
                    "role": path["role"],
                    "query": path["query"],
                    "nodes": status["committed_nodes"],
                    "seconds": status["model_seconds"],
                },
                flush=True,
            )
        status["status"] = "SOURCE_ACQUIRED" if source else "APPLICATION_ACQUIRED"
        persist()
    except BaseException as exc:
        status.update(
            status="RESOURCE_BOUNDARY" if isinstance(exc, TimeoutError) else "ERROR",
            error=repr(exc),
        )
        persist()
        write_json(
            out / "error.json",
            {
                "error": repr(exc),
                "status": status,
                "time": now(),
                "automatic_retry": False,
            },
        )
        raise


def freeze_source(out):
    """Seal all 144 source nodes before any application acquisition."""
    m, status = check(out), read_json(out / "run_status.json")
    if (
        status["status"] != "SOURCE_ACQUIRED"
        or status["committed_nodes"] != 144
        or status["in_flight"] is not None
    ):
        raise ValueError("complete source acquisition required")
    hashes = {}
    for p in read_json(out / "00/design.json"):
        for j in range(3):
            path = node_path(out, p["path"], j)
            if p["role"] != "source":
                if path.exists():
                    raise ValueError("application acquired before checkpoint")
                continue
            array_unit(path)
            for f in (path, path.with_suffix(".npz")):
                hashes[str(f.relative_to(out))] = digest(f)
    write_json(
        out / "01/source_barrier.json",
        {
            "status": "SOURCE_CHECKPOINTED",
            "files": hashes,
            "frozen_at": now(),
            "new_fits": m["spec"]["new_fits"],
        },
    )
    write_json(out / "run_status.json", {**status, "status": "SOURCE_CHECKPOINTED"})
    print({"status": "SOURCE_CHECKPOINTED", "files": len(hashes)}, flush=True)


def evaluate(out):
    """Compare fixed quadrature to cached infinitesimal and actual finite actions."""
    m, status = check(out), read_json(out / "run_status.json")
    if status["status"] != "APPLICATION_ACQUIRED" or status["committed_nodes"] != 288:
        raise ValueError("full bounded actual acquisition required")
    spec, comparisons, nodes = m["spec"], [], []
    parent = ROOT / spec["parent"]
    original = {(int(r["unit"]), r["arm"]): r for r in table(parent / "02/actions.csv")}
    operator = {
        (int(r["left_unit"]), int(r["right_unit"]), r["arm"]): r
        for r in table(ROOT / spec["operator_parent"] / "02/actions.csv")
    }
    for p in read_json(out / "00/design.json"):
        banks = []
        for j in range(3):
            path = node_path(out, p["path"], j)
            meta = array_unit(path)
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as a:
                bank = {k: a[k].copy() for k in a.files}
            banks.append(bank)
            for i, arm in enumerate(spec["arms"]):
                nodes.append(
                    {
                        **{
                            k: p[k]
                            for k in (
                                "path",
                                "pair",
                                "wording",
                                "query",
                                "role",
                                "H",
                                "B",
                                "Q",
                            )
                        },
                        "node": j,
                        "t": meta["t"],
                        "arm": arm,
                        **{
                            k: float(bank["mixed_actions"][i, z])
                            for z, k in enumerate(("full", "supported", "unsupported"))
                        },
                    }
                )
        integrals = M.quadrature(np.stack([b["mixed_actions"] for b in banks]))
        for i, arm in enumerate(spec["arms"]):
            left, right = p["left"]["unit"], p["right"]["unit"]
            gradient = float(
                np.sum(
                    (
                        banks[2]["gradient"].astype(float)
                        - banks[0]["gradient"].astype(float)
                    )
                    * banks[0]["delta_h"][i]
                )
                / 2
            )
            native = (
                float(original[left, arm]["odd_action"])
                - float(original[right, arm]["odd_action"])
            ) / 2
            op = operator[left, right, arm]
            r = {
                **{
                    k: p[k]
                    for k in ("path", "pair", "wording", "query", "role", "H", "B", "Q")
                },
                "arm": arm,
                "actual_gradient_halfcontrast": gradient,
                "actual_finite_halfcontrast": native,
                "hm34_predicted": float(op["predicted"]),
                "hm34_operator_error": float(op["operator_error"]),
                "finite_error": gradient - native,
            }
            for rule, value in integrals.items():
                r.update(
                    {
                        rule + "_" + k: float(value[i, z])
                        for z, k in enumerate(("full", "supported", "unsupported"))
                    }
                )
                r[rule + "_gradient_error"] = float(value[i, 0] - gradient)
                r[rule + "_finite_error"] = float(value[i, 0] - native)
                r[rule + "_support_cross"] = float(2 * value[i, 1] * value[i, 2])
            r["quadrature_difference"] = r["simpson_full"] - r["trapezoid_full"]
            comparisons.append(r)
    metrics = []
    for role in ("source", "application"):
        for q in spec["queries"]:
            for arm in spec["arms"]:
                r = [
                    x
                    for x in comparisons
                    if (x["role"], x["query"], x["arm"]) == (role, q, arm)
                ]
                rms = lambda key, rows=r: float(
                    np.sqrt(np.mean([x[key] ** 2 for x in rows]))
                )
                ref = rms("actual_gradient_halfcontrast")
                metrics.append(
                    {
                        "role": role,
                        "query": q,
                        "arm": arm,
                        "paths": len(r),
                        "gradient_rms": ref,
                        "native_rms": rms("actual_finite_halfcontrast"),
                        **{
                            key + "_rms": rms(key)
                            for key in (
                                "simpson_full",
                                "simpson_supported",
                                "simpson_unsupported",
                                "quadrature_difference",
                                "finite_error",
                            )
                        },
                        **{
                            rule + "_relative_gradient_rmse": rms(
                                rule + "_gradient_error"
                            )
                            / ref
                            if ref
                            else None
                            for rule in ("simpson", "trapezoid")
                        },
                        "support_cross_mean": float(
                            np.mean([x["simpson_support_cross"] for x in r])
                        ),
                    }
                )
    write_csv(out / "02/nodes.csv", nodes)
    write_csv(out / "02/path_comparisons.csv", comparisons)
    write_csv(out / "02/metrics.csv", metrics)
    write_json(
        out / "02/summary.json",
        {
            "status": "FINITE_HM35_ACQUIRED_NOT_GLOBAL",
            "paths": 96,
            "node_arm_rows": len(nodes),
            "comparisons": len(comparisons),
            "counts": status,
            "metrics": metrics,
            "new_fits": 0,
        },
    )
    hashes = {
        str(p.relative_to(out)): digest(p)
        for p in (out / "02").iterdir()
        if p.is_file()
    }
    for p in (out / "01").rglob("*"):
        if p.is_file():
            hashes[str(p.relative_to(out))] = digest(p)
    write_json(out / "02/artifact_hashes.json", hashes)
    write_json(out / "run_status.json", {**status, "status": "EVALUATED"})
    print({"status": "EVALUATED", "comparisons": len(comparisons)}, flush=True)


def main():
    """Expose separate preparation, finite acquisition, checkpoint and evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("prepare", "source", "checkpoint", "application", "evaluate")
    )
    parser.add_argument("--out", type=Path, default=OUT)
    a = parser.parse_args()
    with collector_guard(a.out):
        if a.stage == "prepare":
            prepare(a.out)
        elif a.stage == "checkpoint":
            freeze_source(a.out)
        elif a.stage == "evaluate":
            evaluate(a.out)
        else:
            calls(a.out, a.stage == "source")


if __name__ == "__main__":
    main()
