#!/usr/bin/env python3
"""HM42 source-first actual full-state derivative cut and signed physical actions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROCESS_STARTED = time.monotonic()

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_mixed_query_path as MIX
from grain_research.construction import math_query_context as CONTEXT
from grain_research.construction import math_readout_curvature as CURVE
from grain_research.construction import math_text_query_route_cut as R
from grain_research.vector_mechanisms.direct_traits import native_readout
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts import run_grain_math_mixed_query_path as OLD
from scripts.run_grain_math_position_support import check, now
from scripts.run_grain_math_response import write_csv, write_json
from scripts.run_grain_math_span_binding import hidden_tensor, load_observer
from scripts.run_grain_math_target_active_support import load, save
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_text_query_route_cut_v1.json"
PLAN = ROOT / "docs/GRAIN_TEXT_QUERY_ROUTE_CUT_PLAN_V1.md"
OUT = ROOT / "results/grain_math_response_v1/text_query_route_cut_r1"
CODE = (
    "grain_research/construction/math_text_query_route_cut.py",
    "scripts/run_grain_math_text_query_route_cut.py",
    "scripts/verify_grain_math_text_query_route_cut.py",
    "tests/test_grain_math_text_query_route_cut.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)


def node_file(out, path):
    """Bind each raw cut to its original unique midpoint address."""
    return out / "01/nodes" / f"path_{path:03d}_node_1.json"


def cpu_charge(out, started=PROCESS_STARTED):
    """Charge process imports and stage time without modifying frozen result objects."""
    s = read_json(out / "run_status.json")
    s["analysis_seconds"] += time.monotonic() - started
    write_json(out / "run_status.json", s)
    if s["analysis_seconds"] > 180:
        s.update(
            status="STOPPED_FAILURE_OR_BUDGET", error="cumulative CPU cap exceeded"
        )
        write_json(out / "run_status.json", s)
        raise TimeoutError("fixed180-second cumulative CPU cap reached")


def prepare(out):
    """Freeze all ancestry, code and exact mask addresses before acquisition."""
    if out.exists():
        raise FileExistsError("preserve previous packet")
    spec = read_json(SPEC)
    R.assert_scope(spec)
    previous = ROOT / spec["previous_packet"]
    pm = check(previous)
    vf = ROOT / spec["previous_verification"]
    if read_json(vf)["status"] != "VERIFIED_FINITE_HM41_NOT_GLOBAL":
        raise ValueError("verified HM41 required")
    files = dict(pm["files"])
    for name, sha in read_json(previous / "02/artifact_hashes.json").items():
        if digest(previous / name) != sha:
            raise ValueError("changed prior artifact " + name)
        files[str((previous / name).relative_to(ROOT))] = sha
    files[str(vf.relative_to(ROOT))] = digest(vf)
    design = read_json(ROOT / spec["parent"] / "00/design.json")
    if len(design) != 96 or sum(p["role"] == "source" for p in design) != 48:
        raise ValueError("all original96midpoints required")
    for p in design:
        R.route_masks(p["left"])
        if p["left"]["input_ids"] != p["right"]["input_ids"]:
            raise ValueError("original ordered input token identity required")
    for folder in ("00/code", "01/nodes", "02"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    for name in CODE:
        files[name] = digest(ROOT / name)
        dst = out / "00/code" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / name).read_bytes())
    write_json(out / "00/design.json", design)
    mm = read_json(ROOT / spec["mixed_parent"] / "00/manifest.json")
    write_json(
        out / "00/manifest.json",
        {
            "files": files,
            "inputs": {"00/design.json": digest(out / "00/design.json")},
            "spec": spec,
            "alphabet": mm["alphabet"],
            "frozen_at": now(),
            "new_fits": 0,
        },
    )
    write_json(out / "run_status.json", {**MIX.new_status(), "analysis_seconds": 0.0})
    cpu_charge(out)
    print({"status": "PREPARED", "paths": len(design)}, flush=True)


def observe(observer, row, m, receiver, donors, journal):
    """One unmodified forward after original leaf patch, ten complete cut derivatives."""
    import torch

    hp, pp = row["spans"]["A_property"]["positions"], row["early_query_positions"]
    CURVE.validate_layout(row, donors["h"].shape, donors["p"].shape)
    masks, slots = R.route_masks(row), {}

    def patch(_module, _args, output):
        leaf, audit = CONTEXT.replace_leaf(
            hidden_tensor(output), hp, pp, receiver, donors, True
        )
        slots.update(leaf=leaf, audit=audit)
        return (leaf, *output[1:]) if isinstance(output, tuple) else leaf

    def capture(_module, _args, output):
        # A capture-only hook leaves the forward state unchanged.
        slots["cut"] = hidden_tensor(output)

    handles = [
        observer.model.model.layers[m["spec"]["layer"]].register_forward_hook(patch),
        observer.model.model.layers[m["spec"]["cut_layer"]].register_forward_hook(
            capture
        ),
    ]
    try:
        device = next(observer.model.parameters()).device
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
        full_logits = output.logits[0, -1]
        logits = full_logits[m["alphabet"]["token_ids"]].float()
        jac, cut_grad = R.route_jacobians(
            logits, slots["cut"], slots["leaf"], hp, masks, journal
        )
        values = torch.tensor(
            m["alphabet"]["values"], dtype=torch.float32, device=device
        )
        score = float((torch.softmax(logits, 0) * values).sum().detach().cpu())
        arrays = {
            "logits": logits.detach().cpu().numpy().copy(),
            "route_jacobian": jac,
            "cut_gradients": cut_grad,
            "masks": masks,
            "cut_state": slots["cut"][0].detach().cpu().numpy().copy(),
        }
        return (
            arrays,
            native_readout(full_logits.detach().float().cpu().numpy(), m["alphabet"]),
            score,
            slots["audit"],
        )
    finally:
        for h in handles:
            h.remove()


def acquire(out, source):
    """Finite source/application stage; keep failed node data and uncertain call state."""
    m, s = check(out), read_json(out / "run_status.json")
    spec = m["spec"]
    barrier = out / "01/source_barrier.json"
    R.require_stage(s, source, barrier.exists())
    source_sha = None if source else digest(barrier)
    if not source:
        for name, sha in read_json(barrier)["files"].items():
            if digest(out / name) != sha:
                raise ValueError("changed source checkpoint")
    previous = s["model_seconds"]
    log = out / "01" / ("source_calls.jsonl" if source else "application_calls.jsonl")

    def persist():
        s["model_seconds"] = previous + time.monotonic() - PROCESS_STARTED
        write_json(out / "run_status.json", s)

    def journal(kind, phase):
        nonlocal s
        if phase == "attempted":
            if (
                previous + time.monotonic() - PROCESS_STARTED
                >= spec["maximum_model_seconds"]
            ):
                raise TimeoutError("fixed1200-second cumulative model cap reached")
            s = MIX.reserve(s, kind, spec[f"maximum_{kind}_calls"])
        else:
            s[f"{kind}_completed"] += 1
        with log.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "at": now(),
                        "path": s["in_flight"]["path"],
                        "kind": kind,
                        "phase": phase,
                        "number": s[f"{kind}_{phase}"],
                    }
                )
                + "\n"
            )
        persist()

    try:
        observer = load_observer()
        if (
            next(observer.model.parameters()).device.type != spec["device"]
            or observer.model.config._attn_implementation != spec["attention_backend"]
            or any(p.requires_grad for p in observer.model.parameters())
            or len(observer.model.model.layers) <= spec["cut_layer"]
        ):
            raise ValueError("unchanged frozen sequential eager MPS decoder required")
        write_json(
            out
            / "01"
            / ("source_observer.json" if source else "application_observer.json"),
            observer.manifest(),
        )
        mixed = ROOT / spec["mixed_parent"]
        mm = read_json(mixed / "00/manifest.json")
        for path in read_json(out / "00/design.json"):
            if (path["role"] == "source") != source:
                continue
            dst = node_file(out, path["path"])
            if dst.exists() or dst.with_suffix(".npz").exists():
                raise ValueError("existing observed midpoint cannot be overwritten")
            s.update(
                status="SOURCE_CALLS" if source else "APPLICATION_CALLS",
                in_flight={"path": path["path"], "node": 1, "started_at": now()},
            )
            persist()
            meta, old = load(node_file(ROOT / spec["parent"], path["path"]))
            if meta["address"] != path or meta["node"] != 1 or meta["t"] != 0.5:
                raise ValueError("identical inherited midpoint required")
            receiver, donors, fixed = OLD.inputs(mixed, path, mm)
            donors["p"] = MIX.interpolate(fixed["p0"], fixed["p1"], 0.5)
            donor_meta, original_donors = load(node_file(mixed, path["path"]))
            if (
                donor_meta["address"] != path
                or donor_meta["node"] != 1
                or donor_meta["t"] != 0.5
            ):
                raise ValueError("identical inherited donor midpoint required")
            for k in ("h", "p", "q", "delta_h"):
                if not np.array_equal(
                    (fixed if k == "delta_h" else donors)[k], original_donors[k]
                ):
                    raise ValueError("changed physical donor " + k)
            arrays, native, score, audit = observe(
                observer, path["left"], m, receiver, donors, journal
            )
            replay = {
                "logits_max_error": float(
                    np.max(abs(arrays["logits"].astype(float) - old["logits"]))
                ),
                "score_error": abs(score - meta["score"]),
                "h_jacobian_max_error": float(
                    np.max(
                        abs(
                            arrays["route_jacobian"].astype(float).sum(axis=0)
                            - old["h_jacobian"]
                        )
                    )
                ),
            }
            good = (
                native == meta["native"]
                and max(replay.values()) <= spec["absolute_tolerance"]
                and max(audit.values()) == 0
            )
            save(
                dst,
                {
                    "address": path,
                    "node": 1,
                    "t": 0.5,
                    "native": native,
                    "score": score,
                    "audit": audit,
                    "replay": replay,
                    "replay_valid": bool(good),
                    "source_sha256": source_sha,
                    "groups": spec["groups"],
                    "cut_layer": spec["cut_layer"],
                    "acquired_at": now(),
                },
                arrays,
            )
            if not good:
                raise ValueError(
                    "original1e-8 replay failed; raw node preserved: " + str(replay)
                )
            s["committed_nodes"] += 1
            s["in_flight"] = None
            persist()
            if s["model_seconds"] > spec["maximum_model_seconds"]:
                raise TimeoutError("model cap exceeded during node save")
            if s["committed_nodes"] % 8 == 0:
                print(
                    {
                        "nodes": s["committed_nodes"],
                        "model_seconds": s["model_seconds"],
                        **replay,
                    },
                    flush=True,
                )
        s["status"] = "SOURCE_ACQUIRED" if source else "APPLICATION_ACQUIRED"
        persist()
    except Exception as exc:
        s.update(status="STOPPED_FAILURE_OR_BUDGET", error=repr(exc))
        persist()
        raise


def seal(out):
    """Seal actual source routes, not application-selected reading coefficients."""
    check(out)
    s = read_json(out / "run_status.json")
    if s["status"] != "SOURCE_ACQUIRED" or s["committed_nodes"] != 48:
        raise ValueError("complete48source capture required")
    files = {
        str(p.relative_to(out)): digest(p)
        for p in (out / "01").rglob("*")
        if p.is_file() and not p.name.startswith("._")
    }
    write_json(
        out / "01/source_barrier.json",
        {
            "files": files,
            "source_nodes": 48,
            "application_nodes_loaded": 0,
            "frozen_at": now(),
        },
    )
    s["status"] = "SOURCE_CHECKPOINTED"
    write_json(out / "run_status.json", s)
    cpu_charge(out)
    print({"status": "SOURCE_CHECKPOINTED", "source_nodes": 48}, flush=True)


def analyze(out):
    """Compose every cut route with original C/J_P and retain all signed contrasts."""
    m, s = check(out), read_json(out / "run_status.json")
    spec = m["spec"]
    if s["status"] != "APPLICATION_ACQUIRED" or s["committed_nodes"] != 96:
        raise ValueError("complete96captures required")
    for kind, count in (("forward", 96), ("first", 3840), ("second", 0)):
        if s[f"{kind}_attempted"] != count or s[f"{kind}_completed"] != count:
            raise ValueError("complete actual call journal required")
    source_sha = digest(out / "01/source_barrier.json")
    words, desc = [], []
    maximum = 0.0
    for path in read_json(out / "00/design.json"):
        meta, a = load(node_file(out, path["path"]))
        if (
            meta["address"] != path
            or not meta["replay_valid"]
            or (path["role"] == "application" and meta["source_sha256"] != source_sha)
        ):
            raise ValueError("changed address, failed replay or missing source barrier")
        _, old = load(node_file(ROOT / spec["parent"], path["path"]))
        routes = R.link_routes(
            a["route_jacobian"], old["delta_h"], old["hessian"], old["p_jacobian"]
        )
        maximum = max(
            maximum,
            R.require_replay(
                routes.sum(axis=0), old["link"], spec["absolute_tolerance"]
            ),
        )
        rows, cut_roundoff = R.action_rows(
            path, routes, old, spec["groups"], spec["arms"]
        )
        words.extend(rows)
        save(
            out / "02" / f"path_{path['path']:03d}.json",
            {
                "path": path["path"],
                "query": path["query"],
                "role": path["role"],
                "source_sha256": source_sha,
            },
            {"routes": routes, "cut_roundoff": cut_roundoff},
        )
        d = {
            "path": path["path"],
            "query": path["query"],
            "role": path["role"],
            "cell": rows[0]["cell"],
            "wording": path["wording"],
            **meta["replay"],
        }
        for gi, group in enumerate(R.GROUPS):
            d[group + "_positions"] = int(a["masks"][gi].sum())
            d[group + "_jacobian_norm"] = float(
                np.linalg.norm(a["route_jacobian"][gi].astype(float))
            )
            d[group + "_cut_gradient_norm"] = float(
                np.linalg.norm(a["cut_gradients"][:, a["masks"][gi]].astype(float))
            )
        desc.append(d)
    paired, metrics = [], []
    for query in spec["queries"]:
        for role in ("source", "application"):
            for arm in spec["arms"]:
                w = [
                    r
                    for r in words
                    if (r["query"], r["role"], r["arm"]) == (query, role, arm)
                ]
                p = []
                for cell in range(8):
                    p.extend(
                        R.pair_rows(
                            sorted(
                                [r for r in w if r["cell"] == cell],
                                key=lambda r: r["wording"],
                            ),
                            spec["wordings"],
                        )
                    )
                paired.extend(p)
                for measure in ("word", "mean", "contrast"):
                    selected = (
                        w
                        if measure == "word"
                        else [r for r in p if r["measure"] == measure]
                    )
                    metrics.append(
                        {
                            "query": query,
                            "role": role,
                            "arm": arm,
                            "measure": measure,
                            **R.moments(selected),
                        }
                    )
    write_csv(out / "02/observations.csv", words)
    write_csv(out / "02/paired_actions.csv", paired)
    write_csv(out / "02/action_metrics.csv", metrics)
    write_csv(out / "02/route_descriptors.csv", desc)
    write_json(
        out / "02/summary.json",
        {
            "status": "FINITE_HM42_EVALUATED_NOT_GLOBAL",
            "nodes": 96,
            "observations": len(words),
            "paired_actions": len(paired),
            "metric_groups": len(metrics),
            "maximum_link_replay_error": maximum,
            "new_fits": 0,
            "source_sha256": source_sha,
            "scope": spec["scope"],
        },
    )
    write_json(
        out / "02/artifact_hashes.json",
        {
            str(p.relative_to(out)): digest(p)
            for sub in ("01", "02")
            for p in (out / sub).rglob("*")
            if p.is_file()
            and not p.name.startswith("._")
            and p.name != "artifact_hashes.json"
        },
    )
    s["status"] = "EVALUATED"
    write_json(out / "run_status.json", s)
    cpu_charge(out)
    print(read_json(out / "02/summary.json"), flush=True)


def main():
    """Explicit finite stages; do not auto-resume or retry a failed call."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("prepare", "source", "seal", "application", "analyze")
    )
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    with collector_guard(args.out), threadpool_limits(limits=1):
        if args.stage == "prepare":
            prepare(args.out)
        elif args.stage in ("source", "application"):
            acquire(args.out, args.stage == "source")
        elif args.stage == "seal":
            seal(args.out)
        else:
            analyze(args.out)


if __name__ == "__main__":
    main()
