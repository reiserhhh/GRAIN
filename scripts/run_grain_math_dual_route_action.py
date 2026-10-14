#!/usr/bin/env python3
"""HM43 source-first fixed-weight observable route acquisition and analysis."""

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

from grain_research.construction import math_dual_route_action as D
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

SPEC = ROOT / "configs/grain_math_dual_route_action_v1.json"
PLAN = ROOT / "docs/GRAIN_DUAL_ROUTE_ACTION_PLAN_V1.md"
OUT = ROOT / "results/grain_math_response_v1/dual_route_action_r1"
CODE = (
    "grain_research/construction/math_dual_route_action.py",
    "scripts/run_grain_math_dual_route_action.py",
    "scripts/verify_grain_math_dual_route_action.py",
    "tests/test_grain_math_dual_route_action.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)


def weight_file(out, path):
    """One immutable diagnostic weight pair for each inherited midpoint."""
    return out / "00/weights" / f"path_{path:03d}.json"


def prepare(out):
    """Freeze all96 weights before acquisition; disclose inherited application factors."""
    if out.exists():
        raise FileExistsError("preserve existing packet")
    spec = read_json(SPEC)
    D.assert_scope(spec)
    previous, audit = ROOT / spec["previous_packet"], ROOT / spec["previous_audit"]
    pm = check(previous)
    if read_json(previous / "run_status.json")["status"] != "STOPPED_FAILURE_OR_BUDGET":
        raise ValueError("preserved HM42 stopped boundary required")
    if (
        read_json(audit / "summary.json")["status"]
        != "VERIFIED_NONPROMOTING_PARTIAL_SOURCE_HM42"
    ):
        raise ValueError("nonpromoting HM42 partial audit required")
    files = dict(pm["files"])
    files.update(read_json(audit / "manifest.json")["files"])
    for name, sha in files.items():
        if digest(ROOT / name) != sha:
            raise ValueError("changed historical input " + name)
    for name, sha in read_json(audit / "artifact_hashes.json").items():
        if digest(audit / name) != sha:
            raise ValueError("changed historical audit " + name)
        files[str((audit / name).relative_to(ROOT))] = sha
    for p in (
        previous / "00/manifest.json",
        previous / "run_status.json",
        audit / "manifest.json",
        audit / "artifact_hashes.json",
    ):
        files[str(p.relative_to(ROOT))] = digest(p)
    design = read_json(ROOT / spec["parent"] / "00/design.json")
    if (
        len(design) != 96
        or [p["role"] for p in design] != ["source"] * 48 + ["application"] * 48
    ):
        raise ValueError("original96 ordered source/application paths required")
    for folder in ("00/code", "00/weights", "01/nodes", "02"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    for path in design:
        D.route_masks(path["left"])
        if path["left"]["input_ids"] != path["right"]["input_ids"]:
            raise ValueError("identical original tokens required")
        meta, old = load(node_file(ROOT / spec["parent"], path["path"]))
        if meta["address"] != path or meta["node"] != 1 or meta["t"] != 0.5:
            raise ValueError("original midpoint required")
        w64, w32 = D.dual_weights(old["hessian"], old["p_jacobian"], old["delta_p"])
        save(
            weight_file(out, path["path"]),
            {"address": path, "inherited_local_diagnostic": True},
            {"weight64": w64, "weight32": w32},
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
    mm = read_json(ROOT / spec["mixed_parent"] / "00/manifest.json")
    write_json(
        out / "00/manifest.json",
        {
            "files": files,
            "inputs": inputs,
            "spec": spec,
            "alphabet": mm["alphabet"],
            "frozen_at": now(),
            "new_fits": 0,
            "application_weights_prepared": True,
            "weight_scope": "all inherited local diagnostic factors, not source-only learned weights",
        },
    )
    write_json(out / "run_status.json", {**MIX.new_status(), "analysis_seconds": 0.0})
    cpu_charge(out, PROCESS_STARTED)
    print(
        {"status": "PREPARED", "weights": 96, "application_weights_prepared": True},
        flush=True,
    )


def observe(observer, row, m, receiver, donors, weight32, journal):
    """Replay original forward and differentiate one fixed scalar at a complete cut."""
    import torch

    hp, pp = row["spans"]["A_property"]["positions"], row["early_query_positions"]
    CURVE.validate_layout(row, donors["h"].shape, donors["p"].shape)
    masks, slots = D.route_masks(row), {}

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
        covectors, downstream = D.route_covectors(
            logits, slots["cut"], slots["leaf"], hp, masks, weight32, journal
        )
        values = torch.tensor(
            m["alphabet"]["values"], dtype=torch.float32, device=device
        )
        score = float((torch.softmax(logits, 0) * values).sum().detach().cpu())
        arrays = {
            "logits": logits.detach().cpu().numpy().copy(),
            "route_covectors": covectors,
            "cut_gradient": downstream,
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
    """Source barrier, actual attempted/completed accounting and saved failures."""
    m, s = check(out), read_json(out / "run_status.json")
    spec, barrier = m["spec"], out / "01/source_barrier.json"
    D.require_stage(s, source, barrier.exists())
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
            raise ValueError("unchanged frozen sequential eager MPS reader required")
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
                raise ValueError("cannot overwrite an acquired node")
            s.update(
                status="SOURCE_CALLS" if source else "APPLICATION_CALLS",
                in_flight={"path": path["path"], "node": 1, "started_at": now()},
            )
            persist()
            meta, old = load(node_file(ROOT / spec["parent"], path["path"]))
            wm, weights = load(weight_file(out, path["path"]))
            if meta["address"] != path or wm["address"] != path or meta["t"] != 0.5:
                raise ValueError("changed frozen input address")
            receiver, donors, fixed = OLD.inputs(mixed, path, mm)
            donors["p"] = MIX.interpolate(fixed["p0"], fixed["p1"], 0.5)
            dm, original = load(node_file(mixed, path["path"]))
            if dm["address"] != path or dm["node"] != 1 or dm["t"] != 0.5:
                raise ValueError("changed donor midpoint")
            for k in ("h", "p", "q", "delta_h"):
                if not np.array_equal(
                    (fixed if k == "delta_h" else donors)[k], original[k]
                ):
                    raise ValueError("changed donor " + k)
            arrays, native, score, audit = observe(
                observer,
                path["left"],
                m,
                receiver,
                donors,
                weights["weight32"],
                journal,
            )
            arrays.update(weights)
            accounts = D.scalar_accounts(
                arrays["route_covectors"], weights["weight64"], weights["weight32"], old
            )
            replay = {
                "logits_max_error": float(
                    np.max(abs(arrays["logits"].astype(float) - old["logits"]))
                ),
                "score_error": abs(score - meta["score"]),
                "physical_action_max_error": float(
                    np.max(abs(accounts["raw_link"] - accounts["link"]))
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
                    "weight_scope": m["weight_scope"],
                },
                arrays,
            )
            if not good:
                raise ValueError(
                    "original physical1e-8 gate failed; raw node saved: " + str(replay)
                )
            s["committed_nodes"] += 1
            s["in_flight"] = None
            persist()
            if s["model_seconds"] > spec["maximum_model_seconds"]:
                raise TimeoutError("model cap exceeded during save")
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
    """Seal new source acquisitions, while disclosing historical application weights."""
    check(out)
    s = read_json(out / "run_status.json")
    if s["status"] != "SOURCE_ACQUIRED" or s["committed_nodes"] != 48:
        raise ValueError("complete48 source acquisition required")
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
            "new_application_nodes_acquired": 0,
            "application_weights_prepared": True,
            "frozen_at": now(),
        },
    )
    s["status"] = "SOURCE_CHECKPOINTED"
    write_json(out / "run_status.json", s)
    cpu_charge(out, PROCESS_STARTED)
    print({"status": "SOURCE_CHECKPOINTED", "source_nodes": 48}, flush=True)


def analyze(out):
    """Keep three raw routes, distinct corrections, pair contrasts and signed crosses."""
    m, s = check(out), read_json(out / "run_status.json")
    spec = m["spec"]
    if s["status"] != "APPLICATION_ACQUIRED" or s["committed_nodes"] != 96:
        raise ValueError("complete96 actual acquisitions required")
    for kind, n in (("forward", 96), ("first", 384), ("second", 0)):
        if s[f"{kind}_attempted"] != n or s[f"{kind}_completed"] != n:
            raise ValueError("actual finite call counts required")
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
            raise ValueError("changed address/replay/barrier")
        _, old = load(node_file(ROOT / spec["parent"], path["path"]))
        accounts = D.scalar_accounts(
            a["route_covectors"], a["weight64"], a["weight32"], old
        )
        maximum = max(maximum, D.require_replay(accounts["raw_link"], accounts["link"]))
        rows = D.action_rows(path, accounts, spec["arms"])
        words.extend(rows)
        save(
            out / "02" / f"path_{path['path']:03d}.json",
            {"path": path["path"], "source_sha256": source_sha},
            accounts,
        )
        d = {k: path[k] for k in ("path", "query", "role", "wording")}
        d.update(cell=rows[0]["cell"], **meta["replay"])
        for gi, group in enumerate(D.GROUPS):
            d[group + "_positions"] = int(a["masks"][gi].sum())
            d[group + "_covector_norm"] = float(
                np.linalg.norm(a["route_covectors"][gi].astype(float))
            )
            d[group + "_cut_gradient_norm"] = float(
                np.linalg.norm(a["cut_gradient"][a["masks"][gi]].astype(float))
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
                        D.pair_rows(
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
                            **D.moments(selected),
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
            "status": "FINITE_HM43_EVALUATED_NOT_GLOBAL",
            "nodes": 96,
            "observations": len(words),
            "paired_actions": len(paired),
            "metric_groups": len(metrics),
            "maximum_physical_action_replay_error": maximum,
            "new_fits": 0,
            "source_sha256": source_sha,
            "application_weights_prepared": True,
            "HM42_full_jacobian_accepted": False,
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
    """Finite explicit stages; a failed stage cannot be automatically replayed."""
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
