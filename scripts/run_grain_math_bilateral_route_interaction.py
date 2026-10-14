#!/usr/bin/env python3
"""HM44 actual source-first acquisition of two-leg signed route interactions."""

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

from grain_research.construction import math_bilateral_route_interaction as B
from grain_research.construction import math_mixed_query_path as MIX
from grain_research.construction import math_query_context as CONTEXT
from grain_research.construction import math_readout_curvature as CURVE
from grain_research.vector_mechanisms.direct_traits import native_readout
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts import run_grain_math_mixed_query_path as OLD
from scripts.run_grain_math_position_support import check, now
from scripts.run_grain_math_response import write_csv, write_json
from scripts.run_grain_math_span_binding import hidden_tensor, load_observer
from scripts.run_grain_math_target_active_support import load, save
from scripts.run_grain_math_text_query_route_cut import cpu_charge, node_file
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_bilateral_route_interaction_v1.json"
PLAN = ROOT / "docs/GRAIN_BILATERAL_ROUTE_INTERACTION_PLAN_V1.md"
OUT = ROOT / "results/grain_math_response_v1/bilateral_route_interaction_r1"
CODE = (
    "grain_research/construction/math_bilateral_route_interaction.py",
    "scripts/run_grain_math_bilateral_route_interaction.py",
    "scripts/verify_grain_math_bilateral_route_interaction.py",
    "tests/test_grain_math_bilateral_route_interaction.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)


def factor_file(out, path):
    """One immutable inherited C/d/e bundle for each original midpoint."""
    return out / "00/factors" / f"path_{path:03d}.json"


def prepare(out):
    """Freeze24 predeclared corners and all diagnostic factors before any call."""
    if out.exists():
        raise FileExistsError("preserve existing finite packet")
    spec = read_json(SPEC)
    B.assert_scope(spec)
    previous = ROOT / spec["previous_packet"]
    pm = check(previous)
    vf = ROOT / spec["previous_verification"]
    if read_json(vf)["status"] != "VERIFIED_FINITE_HM43_NOT_GLOBAL":
        raise ValueError("verified complete HM43 required")
    files = dict(pm["files"])
    for name, sha in read_json(previous / "02/artifact_hashes.json").items():
        if digest(previous / name) != sha:
            raise ValueError("changed HM43 output " + name)
        files[str((previous / name).relative_to(ROOT))] = sha
    for name, sha in pm["inputs"].items():
        files[str((previous / name).relative_to(ROOT))] = sha
    for p in (vf, previous / "00/manifest.json", previous / "02/artifact_hashes.json"):
        files[str(p.relative_to(ROOT))] = digest(p)
    design = [
        p
        for p in read_json(ROOT / spec["parent"] / "00/design.json")
        if p["path"] in B.PATHS
    ]
    if [p["path"] for p in design] != list(B.PATHS) or [p["role"] for p in design] != [
        "source"
    ] * 12 + ["application"] * 12:
        raise ValueError("original24 ordered corners required")
    for sub in ("00/code", "00/factors", "01/nodes", "02"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for p in design:
        B.route_masks(p["left"])
        if (
            p["H"] * 4 + p["B"] * 2 + p["Q"] not in (0, 7)
            or p["left"]["input_ids"] != p["right"]["input_ids"]
        ):
            raise ValueError("original paired corner tokens required")
        meta, old = load(node_file(ROOT / spec["parent"], p["path"]))
        if meta["address"] != p or meta["node"] != 1 or meta["t"] != 0.5:
            raise ValueError("unchanged original midpoint required")
        save(
            factor_file(out, p["path"]),
            {"address": p, "local_diagnostic": True},
            {k: old[k] for k in ("hessian", "delta_h", "delta_p")},
        )
    for name in CODE:
        files[name] = digest(ROOT / name)
        dst = out / "00/code" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / name).read_bytes())
    write_json(out / "00/design.json", design)
    inputs = {
        str(p.relative_to(out)): digest(p)
        for p in (out / "00").rglob("*")
        if p.is_file() and not p.name.startswith("._")
    }
    write_json(
        out / "00/manifest.json",
        {
            "files": files,
            "inputs": inputs,
            "spec": spec,
            "alphabet": pm["alphabet"],
            "frozen_at": now(),
            "new_fits": 0,
            "application_factors_prepared": True,
            "factor_scope": "inherited local diagnostic C/d/e, not source-learned cold weights",
        },
    )
    write_json(out / "run_status.json", {**MIX.new_status(), "analysis_seconds": 0.0})
    cpu_charge(out, PROCESS_STARTED)
    print(
        {
            "status": "PREPARED",
            "nodes": 24,
            "source": 12,
            "application_factors_prepared": True,
        },
        flush=True,
    )


def observe(observer, row, m, receiver, donors, journal):
    """One original patched forward; masked leaf gradients yield both patch legs."""
    import torch

    hp, pp = row["spans"]["A_property"]["positions"], row["early_query_positions"]
    CURVE.validate_layout(row, donors["h"].shape, donors["p"].shape)
    masks, slots = B.route_masks(row), {}

    def patch(_module, _args, output):
        leaf, audit = CONTEXT.replace_leaf(
            hidden_tensor(output), hp, pp, receiver, donors, True
        )
        slots.update(leaf=leaf, audit=audit)
        return (leaf, *output[1:]) if isinstance(output, tuple) else leaf

    def capture(_module, _args, output):
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
        hj, pj, down = B.bilateral_jacobians(
            logits, slots["cut"], slots["leaf"], hp, pp, masks, journal
        )
        values = torch.tensor(
            m["alphabet"]["values"], dtype=torch.float32, device=device
        )
        score = float((torch.softmax(logits, 0) * values).sum().detach().cpu())
        return (
            {
                "logits": logits.detach().cpu().numpy().copy(),
                "h_routes": hj,
                "p_routes": pj,
                "cut_gradients": down,
                "masks": masks,
                "cut_state": slots["cut"][0].detach().cpu().numpy().copy(),
            },
            native_readout(full_logits.detach().float().cpu().numpy(), m["alphabet"]),
            score,
            slots["audit"],
        )
    finally:
        for h in handles:
            h.remove()


def acquire(out, source):
    """Save each actual raw node before replay gates; never retry a failed/uncertain call."""
    m, s = check(out), read_json(out / "run_status.json")
    spec, barrier = m["spec"], out / "01/source_barrier.json"
    B.require_stage(s, source, barrier.exists())
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
            if previous + time.monotonic() - PROCESS_STARTED >= 600:
                raise TimeoutError("fixed600-second model cap")
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
            raise ValueError("unchanged frozen eager MPS sequential reader required")
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
                raise ValueError("existing raw node cannot be overwritten")
            s.update(
                status="SOURCE_CALLS" if source else "APPLICATION_CALLS",
                in_flight={"path": path["path"], "node": 1, "started_at": now()},
            )
            persist()
            meta, old = load(node_file(ROOT / spec["parent"], path["path"]))
            fm, factors = load(factor_file(out, path["path"]))
            if (
                meta["address"] != path
                or fm["address"] != path
                or meta["node"] != 1
                or meta["t"] != 0.5
            ):
                raise ValueError("changed original address")
            for k in factors:
                if not np.array_equal(factors[k], old[k]):
                    raise ValueError("changed diagnostic factor " + k)
            receiver, donors, fixed = OLD.inputs(mixed, path, mm)
            donors["p"] = MIX.interpolate(fixed["p0"], fixed["p1"], 0.5)
            dm, original = load(node_file(mixed, path["path"]))
            if dm["address"] != path or dm["node"] != 1 or dm["t"] != 0.5:
                raise ValueError("changed physical donor address")
            for k in ("h", "p", "q", "delta_h"):
                if not np.array_equal(
                    (fixed if k == "delta_h" else donors)[k], original[k]
                ):
                    raise ValueError("changed physical donor " + k)
            arrays, native, score, audit = observe(
                observer, path["left"], m, receiver, donors, journal
            )
            joint = B.joint_actions(
                arrays["h_routes"],
                arrays["p_routes"],
                factors["delta_h"],
                factors["delta_p"],
                factors["hessian"],
            )
            accounts = B.scalar_accounts(joint, old)
            _, hm43 = load(node_file(ROOT / spec["previous_packet"], path["path"]))
            hm43_actions = (
                hm43["route_covectors"].astype(float).reshape(3, -1)
                @ factors["delta_h"].reshape(2, -1).T
            )
            replay = {
                "logits_max_error": float(
                    np.max(abs(arrays["logits"].astype(float) - old["logits"]))
                ),
                "score_error": abs(score - meta["score"]),
                "physical_action_max_error": float(
                    np.max(abs(accounts["raw_link"] - accounts["link"]))
                ),
                "H_margin_HM43_max_error": float(
                    np.max(abs(joint.sum(2).T - hm43_actions))
                ),
                "causal_P_body_max": float(np.max(abs(arrays["p_routes"][0]))),
            }
            coordinate_errors = {
                "H_max": float(
                    np.max(
                        abs(arrays["h_routes"].astype(float).sum(0) - old["h_jacobian"])
                    )
                ),
                "P_max": float(
                    np.max(
                        abs(arrays["p_routes"].astype(float).sum(0) - old["p_jacobian"])
                    )
                ),
            }
            good = (
                native == meta["native"]
                and max(replay.values()) <= 1e-8
                and replay["causal_P_body_max"] == 0
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
                    "coordinate_errors_reported_not_acceptance": coordinate_errors,
                    "replay_valid": bool(good),
                    "source_sha256": source_sha,
                    "acquired_at": now(),
                    "factor_scope": m["factor_scope"],
                },
                arrays,
            )
            if not good:
                raise ValueError(
                    "original physical gate failed; raw node retained: " + str(replay)
                )
            s["committed_nodes"] += 1
            s["in_flight"] = None
            persist()
            if s["model_seconds"] > 600:
                raise TimeoutError("model cap exceeded during node save")
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
    """Seal12 actually acquired source nodes; inherited application factors disclosed."""
    check(out)
    s = read_json(out / "run_status.json")
    if s["status"] != "SOURCE_ACQUIRED" or s["committed_nodes"] != 12:
        raise ValueError("complete12 source nodes required")
    files = {
        str(p.relative_to(out)): digest(p)
        for p in (out / "01").rglob("*")
        if p.is_file() and not p.name.startswith("._")
    }
    write_json(
        out / "01/source_barrier.json",
        {
            "files": files,
            "source_nodes": 12,
            "new_application_nodes_acquired": 0,
            "application_factors_prepared": True,
            "frozen_at": now(),
        },
    )
    s["status"] = "SOURCE_CHECKPOINTED"
    write_json(out / "run_status.json", s)
    cpu_charge(out, PROCESS_STARTED)
    print({"status": "SOURCE_CHECKPOINTED", "source": 12}, flush=True)


def analyze(out):
    """Preserve every9-cell signed action and both margins under all fixed contrasts."""
    m, s = check(out), read_json(out / "run_status.json")
    spec = m["spec"]
    if s["status"] != "APPLICATION_ACQUIRED" or s["committed_nodes"] != 24:
        raise ValueError("complete24 actual captures required")
    for kind, n in (("forward", 24), ("first", 960), ("second", 0)):
        if s[f"{kind}_attempted"] != n or s[f"{kind}_completed"] != n:
            raise ValueError("complete actual calls required")
    source_sha = digest(out / "01/source_barrier.json")
    words, desc = [], []
    maximum = 0.0
    for path in read_json(out / "00/design.json"):
        meta, a = load(node_file(out, path["path"]))
        if (
            meta["address"] != path
            or not meta["replay_valid"]
            or meta["source_sha256"]
            != (source_sha if path["role"] == "application" else None)
        ):
            raise ValueError("changed address/replay/source provenance")
        _, old = load(node_file(ROOT / spec["parent"], path["path"]))
        joint = B.joint_actions(
            a["h_routes"], a["p_routes"], old["delta_h"], old["delta_p"], old["hessian"]
        )
        accounts = B.scalar_accounts(joint, old)
        maximum = max(maximum, B.require_replay(accounts["raw_link"], accounts["link"]))
        words.extend(B.action_rows(path, accounts, spec["arms"]))
        save(
            out / "02" / f"path_{path['path']:03d}.json",
            {"path": path["path"], "source_sha256": source_sha},
            {"joint": joint, **accounts},
        )
        d = {k: path[k] for k in ("path", "query", "role", "wording")}
        d.update(
            cell=path["H"] * 4 + path["B"] * 2 + path["Q"],
            **meta["replay"],
            **meta["coordinate_errors_reported_not_acceptance"],
        )
        for gi, g in enumerate(B.GROUPS):
            d[g + "_positions"] = int(a["masks"][gi].sum())
            d[g + "_H_norm"] = float(np.linalg.norm(a["h_routes"][gi].astype(float)))
            d[g + "_P_norm"] = float(np.linalg.norm(a["p_routes"][gi].astype(float)))
        desc.append(d)
    paired, metrics = [], []
    for q in spec["queries"]:
        for role in ("source", "application"):
            for arm in spec["arms"]:
                w = [
                    r
                    for r in words
                    if (r["query"], r["role"], r["arm"]) == (q, role, arm)
                ]
                p = []
                for cell in spec["corners"]:
                    p.extend(
                        B.pair_rows(
                            sorted(
                                [r for r in w if r["cell"] == cell],
                                key=lambda r: r["wording"],
                            ),
                            spec["wordings"],
                        )
                    )
                paired.extend(p)
                for measure in ("word", "mean", "contrast"):
                    rows = (
                        w
                        if measure == "word"
                        else [r for r in p if r["measure"] == measure]
                    )
                    metrics.append(
                        {
                            "query": q,
                            "role": role,
                            "arm": arm,
                            "measure": measure,
                            **B.moments(rows),
                        }
                    )
    for name, rows in (
        ("observations", words),
        ("paired_actions", paired),
        ("action_metrics", metrics),
        ("route_descriptors", desc),
    ):
        write_csv(out / "02" / f"{name}.csv", rows)
    write_json(
        out / "02/summary.json",
        {
            "status": "FINITE_HM44_EVALUATED_NOT_GLOBAL",
            "nodes": 24,
            "observations": len(words),
            "paired_actions": len(paired),
            "metric_groups": len(metrics),
            "maximum_physical_action_replay_error": maximum,
            "source_sha256": source_sha,
            "new_fits": 0,
            "application_factors_prepared": True,
            "HM42_coordinate_accepted": False,
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
    cpu_charge(out, PROCESS_STARTED)
    print(read_json(out / "02/summary.json"), flush=True)


def main():
    """One explicit finite stage at a time; no automatic scientific retries."""
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
