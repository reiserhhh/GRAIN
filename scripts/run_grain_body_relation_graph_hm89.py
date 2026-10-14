#!/usr/bin/env python3
"""Finite body-only relational transport; no target question, answer or fit."""

from __future__ import annotations

import argparse
import itertools
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_body_relation_graph as M
from grain_research.construction import math_cold_evidence as C
from grain_research.vector_mechanisms.navigation import (
    MODEL,
    digest,
    read_json,
    write_json,
)
from grain_research.vector_mechanisms.prefix import capture_prefix
from scripts.run_grain_hierarchical_query_hm88 import table, write
from scripts.run_grain_math_construct_routing import save
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_query_evidence_hm86 import check_hashes, seal, tokenize, unit
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_body_relation_graph_hm89.json"
ARMS = (
    "cold",
    "direct",
    "harmonic",
    "event_harmonic",
    "position",
    "actor_oracle",
    "reporter_oracle",
)
CODE = (
    "configs/grain_body_relation_graph_hm89.json",
    "docs/GRAIN_BODY_RELATION_GRAPH_PLAN_HM89.md",
    "grain_research/construction/math_body_relation_graph.py",
    "scripts/run_grain_body_relation_graph_hm89.py",
    "tests/test_grain_math_body_relation_graph.py",
    "grain_research/vector_mechanisms/prefix.py",
    "grain_research/construction/math_cold_evidence.py",
)


def sentence(event, actor, behavior, frame):
    """Place two equally exposed people in reporting/acting roles without a query."""
    name = "A" if actor > 0 else "B"
    reporter = "B" if name == "A" else "A"
    if frame == "reporter_first":
        return f"On occasion {event}, Person {reporter} recorded that Person {name} {behavior}."
    return f"On occasion {event}, Person {name} {behavior}, according to Person {reporter}."


def material_tokens(body, target, system, tokenizer, actions):
    """Map literal person anchors and analysis-only behavior spans to real tokens."""
    tok = tokenize(body, target, system, tokenizer, True)
    rendered = tok["rendered"]
    offsets = tokenizer(
        rendered, add_special_tokens=False, return_offsets_mapping=True
    )["offset_mapping"]
    ix = sorted(i for g in tok["groups"] for i in g)
    back = {i: j for j, i in enumerate(ix)}
    labels = np.full(len(ix), -1)
    labels[0] = 2
    for match in re.finditer(r"Person ([AB])", body):
        lo = rendered.index(body) + match.start(1)
        span = [i for i in ix if offsets[i][0] < lo + 1 and offsets[i][1] > lo]
        if len(span) != 1:
            raise ValueError("Ambiguous name-token boundary; do not rescue")
        labels[back[span[0]]] = 0 if match[1] == "A" else 1
    action_groups = []
    for event, action in enumerate(actions, start=1):
        line_start = body.index(f"On occasion K{event},")
        lo = rendered.index(body) + body.index(action, line_start)
        hi = lo + len(action)
        action_groups.append(
            [back[i] for i in ix if offsets[i][0] < hi and offsets[i][1] > lo]
        )
    return {
        **tok,
        "body_indices": ix,
        "root_labels": labels.tolist(),
        "action_groups": action_groups,
    }


def prepare(out):
    """Freeze source-only numerical parameters and every new body before capture."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve all completed and partial packets")
    spec = read_json(SPEC)
    source = ROOT / spec["source"]
    ancestry = read_json(source / "00/manifest.json")
    check_hashes(ROOT, ancestry["code"] | ancestry["source"])
    check_hashes(source, ancestry["inputs"])
    check_hashes(source, read_json(source / "01/raw_seal.json"))
    check_hashes(source, read_json(source / "04_verified_completion/output_seal.json"))
    if read_json(source / "verification.json")["cold_arithmetic_and_tables"] != "PASS":
        raise ValueError("HM87 numerical source not verified")
    bundle = read_json(source / "00/bundle.json")
    material = read_json(ROOT / spec["source_material"])
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    rows = []
    for frame, order, target, coordinates in itertools.product(
        spec["frames"], (0, 1), spec["targets"], itertools.product((-1, 1), repeat=4)
    ):
        x, y, g, h = coordinates
        actions = [
            spec["behaviours"][target][int(x > 0)],
            spec["behaviours"][target][int(y > 0)],
        ]
        lines = [
            sentence("K1", g, actions[0], frame),
            sentence("K2", h, actions[1], frame),
            "On occasion K3, Person A and Person B checked a calendar together.",
        ]
        permutation = [0, 1, 2] if order == 0 else [2, 1, 0]
        body = "\n".join(lines[i] for i in permutation)
        rows.append(
            {
                "unit": len(rows),
                "frame": frame,
                "order": order,
                "target": target,
                "coordinates": list(coordinates),
                "event_order": permutation,
                **material_tokens(body, target, material["system"], tokenizer, actions),
            }
        )
    if len(rows) != spec["max_forwards"]:
        raise ValueError("Incomplete fixed material")
    lookup = {(r["frame"], r["order"], r["target"], *r["coordinates"]): r for r in rows}
    bags = 0
    for r in rows:
        x, y, g, h = r["coordinates"]
        for gg, hh in ((-g, h), (g, -h)):
            other = lookup[r["frame"], r["order"], r["target"], x, y, gg, hh]
            if sorted(r["input_ids"]) != sorted(other["input_ids"]):
                raise ValueError("Role flip changed token bag")
            bags += 1
    (out / "00").mkdir(parents=True)
    write(out / "00/rows.json", rows)
    write(out / "00/bundle.json", bundle)
    write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "arms": ARMS,
            "source": {
                str(p.relative_to(ROOT)): digest(p)
                for p in (
                    source / "00/bundle.json",
                    source / "verification.json",
                    source / "01/raw_seal.json",
                    source / "04_verified_completion/output_seal.json",
                )
            },
            "code": {n: digest(ROOT / n) for n in CODE},
            "inputs": seal(out, out / "00"),
            "model": ancestry["model"],
            "paired_token_bags": bags,
            "HM87_failed_prefix_gate_retained": read_json(source / "verification.json")[
                "frozen_prefix_gate"
            ],
            "no_target_query_suffix_or_answer": True,
            "no_application_fitting": True,
        },
    )
    print("PREPARED HM89", len(rows), "body-only units", flush=True)


def collect(out):
    """Capture full body Q at the existing stop boundary with resumable units."""
    manifest, rows = (
        read_json(out / "00/manifest.json"),
        read_json(out / "00/rows.json"),
    )
    check_hashes(ROOT, manifest["code"] | manifest["source"])
    check_hashes(out, manifest["inputs"])
    check_hashes(MODEL, manifest["model"])
    bundle = read_json(out / "00/bundle.json")
    observer = load_observer()
    status_path = out / "01/status.json"
    status = (
        read_json(status_path)
        if status_path.exists()
        else {"completed": 0, "model_seconds": 0.0}
    )
    with collector_guard(out):
        for row in rows:
            path = out / f"01/units/unit_{row['unit']:04d}.json"
            if path.exists():
                unit(out, row["unit"])
                continue
            if status["model_seconds"] >= manifest["spec"]["max_model_seconds"]:
                raise TimeoutError("Finite model budget reached")
            saved = {}

            def full_query(_module, _inputs, output, saved=saved):
                saved["q"] = output[0].detach().float().cpu().numpy().copy()

            handle = observer.model.model.layers[
                17
            ].self_attn.q_proj.register_forward_hook(full_query)
            begin = time.monotonic()
            try:
                raw, trace = capture_prefix(observer, row["input_ids"])
            finally:
                handle.remove()
            elapsed = time.monotonic() - begin
            positions = row["body_indices"]
            queries = saved["q"].reshape(len(row["input_ids"]), 12, -1)
            qbody = np.stack(
                [
                    C.rotate(queries[i], i, bundle["frequencies"], bundle["scaling"])
                    for i in positions
                ]
            )
            raw.pop("boundary")
            save(
                path,
                {
                    "unit": row["unit"],
                    "trace": trace,
                    "seconds": elapsed,
                    "stage": "body_graph",
                },
                {
                    "q_body": qbody,
                    "k": raw["k"],
                    "v": raw["v"],
                    "q": raw["q"],
                    "raw_query": raw["raw_query"],
                    "scaling": raw["scaling"],
                },
            )
            status = {
                "state": "ACTIVE",
                "completed": row["unit"] + 1,
                "model_seconds": status["model_seconds"] + elapsed,
                "calls_this_invocation": observer.calls,
            }
            write_json(status_path, status)
            if status["completed"] % 48 == 0:
                print(status, flush=True)
        status["state"] = "COLLECTED"
        write_json(status_path, status)
    write(out / "01/raw_seal.json", seal(out, out / "01/units"))
    print("COLLECTED", status, flush=True)


def observe(raw, row, bundle):
    """Run all frozen numerical laws; nonoracle arms receive no design actor truth."""
    positions = row["body_indices"]
    k, v = raw["k"][positions], raw["v"][positions]
    labels = np.array(row["root_labels"])
    ids = np.array(
        [
            row["event_order"][
                next(j for j, group in enumerate(row["groups"]) if pos in group)
            ]
            for pos in positions
        ]
    )
    t, self_mass = M.causal_graph(raw["q_body"], k)
    local, _ = M.causal_graph(raw["q_body"], k, ids)
    positional, _ = M.causal_graph(raw["q_body"], k, position_only=True)
    laws = {
        "direct": M.direct_law(t, labels),
        "harmonic": M.boundary_law(t, labels),
        "event_harmonic": M.boundary_law(local, labels),
        "position": M.boundary_law(positional, labels),
    }
    gates = {"cold": np.ones(len(k)), **{name: law[:, 0] for name, law in laws.items()}}
    actors = np.array([row["coordinates"][2], row["coordinates"][3], 0])
    gates["actor_oracle"] = np.where(ids == 2, 0.5, (actors[ids] > 0).astype(float))
    gates["reporter_oracle"] = np.where(ids == 2, 0.5, (actors[ids] < 0).astype(float))
    expanded = np.repeat(v, 12 // v.shape[1], axis=1)
    directions = np.asarray(bundle["directions"])
    values, accounts, errors = np.zeros((len(ARMS), 6)), [], []
    for qi in range(6):
        q = C.rotate(
            bundle["raw_queries"][qi],
            positions[-1] + bundle["gaps"][qi],
            bundle["frequencies"],
            bundle["scaling"],
        )
        logits = np.stack([k[:, h // (12 // k.shape[1])] @ q[h] for h in range(12)])
        p0 = softmax(logits, axis=1)
        for ai, name in enumerate(ARMS):
            obs = M.gated_reading(logits, expanded, gates[name], ids)
            values[ai, qi] = np.sum(obs["reading"] * directions[qi])
            errors.append(
                float(
                    np.max(
                        np.abs(
                            obs["reading"]
                            - np.einsum(
                                "hg,hgd->hd", obs["event_mass"], obs["event_read"]
                            )
                        )
                    )
                )
            )
            accounts.append(
                {
                    "unit": row["unit"],
                    "query": qi,
                    "arm": name,
                    "retained_mass": float(obs["retained_mass"].mean()),
                    "event0": float(obs["event_mass"][:, 0].mean()),
                    "event1": float(obs["event_mass"][:, 1].mean()),
                    "event2": float(obs["event_mass"][:, 2].mean()),
                    "unresolved": float((p0 * laws[name][:, 2][None]).sum(1).mean())
                    if name in laws
                    else 0.0,
                }
            )
    role = []
    for name, law in laws.items():
        for event, group in enumerate(row["action_groups"]):
            sign = float((law[group, 0] - law[group, 1]).mean())
            role.append(
                {
                    "unit": row["unit"],
                    "frame": row["frame"],
                    "order": row["order"],
                    "target": row["target"],
                    "method": name,
                    "event": event,
                    "actor": int(actors[event]),
                    "signed_allocation": sign,
                    "actor_alignment": sign * actors[event],
                    "unresolved": float(law[group, 2].mean()),
                }
            )
    endpoint_error = float(np.max(np.abs(raw["q_body"][-1] - raw["q"])))
    return values, accounts, role, max(errors), endpoint_error, float(self_mass.mean())


def analyze(out):
    """Keep all mechanism signs and author contrasts; never select an accuracy winner."""
    manifest, rows = (
        read_json(out / "00/manifest.json"),
        read_json(out / "00/rows.json"),
    )
    check_hashes(ROOT, manifest["code"] | manifest["source"])
    check_hashes(out, manifest["inputs"])
    check_hashes(out, read_json(out / "01/raw_seal.json"))
    bundle = read_json(out / "00/bundle.json")
    fields = np.zeros((2, 2, 6, 16, len(ARMS), 6))
    accounts, roles, errors, endpoint_errors, diagonal_mass = [], [], [], [], []
    points = list(itertools.product((-1, 1), repeat=4))
    for row in rows:
        value, account, role, error, endpoint_error, self_mass = observe(
            unit(out, row["unit"]), row, bundle
        )
        fi, qi, ci = (
            manifest["spec"]["frames"].index(row["frame"]),
            bundle["targets"].index(row["target"]),
            points.index(tuple(row["coordinates"])),
        )
        fields[fi, row["order"], qi, ci] = value
        accounts.extend(account)
        roles.extend(role)
        errors.append(error)
        endpoint_errors.append(endpoint_error)
        diagonal_mass.append(self_mass)
    pplus, pminus = M.author_laws(np.array(points), manifest["spec"]["epsilon"])
    contrasts = []
    for fi, order, ti, ai, ri in itertools.product(
        range(2), range(2), range(6), range(len(ARMS)), range(6)
    ):
        vals = fields[fi, order, ti, :, ai, ri]
        contrasts.append(
            {
                "frame": manifest["spec"]["frames"][fi],
                "order": order,
                "target": bundle["targets"][ti],
                "readout": bundle["targets"][ri],
                "arm": ARMS[ai],
                "author_law_contrast": float((pplus - pminus) @ vals),
                "is_target": ti == ri,
            }
        )
    summary_roles = []
    for frame, order, method in itertools.product(
        manifest["spec"]["frames"],
        (0, 1),
        ("direct", "harmonic", "event_harmonic", "position"),
    ):
        subset = [
            r
            for r in roles
            if (r["frame"], r["order"], r["method"]) == (frame, order, method)
        ]
        summary_roles.append(
            {
                "frame": frame,
                "order": order,
                "method": method,
                "actor_alignment": float(
                    np.mean([r["actor_alignment"] for r in subset])
                ),
                "unresolved": float(np.mean([r["unresolved"] for r in subset])),
            }
        )
    summary = {
        "status": "EXECUTED_FINITE_HM89_AWAITING_VERIFIER",
        "actual_body_calls": len(rows),
        "model_seconds": read_json(out / "01/status.json")["model_seconds"],
        "event_account_max_error": max(errors),
        "rotary_endpoint_max_error": max(endpoint_errors),
        "mean_removed_self_mass": float(np.mean(diagonal_mass)),
        "role_summary": summary_roles,
        "same_marginal_law_error": float(
            np.max(np.abs(pplus @ np.array(points) - pminus @ np.array(points)))
        ),
        "body_encoder_still_neural": True,
        "graph_not_causal_attribution_or_multilayer_rollout": True,
        "original_scope_complete": False,
    }
    (out / "02").mkdir()
    np.savez_compressed(
        out / "02/projections.npz", fields=fields, pplus=pplus, pminus=pminus
    )
    table(out / "02/probability_accounts.csv", accounts)
    table(out / "02/token_role_transport.csv", roles)
    table(out / "02/author_law_contrasts.csv", contrasts)
    table(out / "02/role_summary.csv", summary_roles)
    write(out / "02/summary.json", summary)
    write(out / "02/output_seal.json", seal(out, out / "02"))
    print(summary, flush=True)


def main():
    """Execute one bounded stage; no automatic next experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "collect", "analyze"))
    parser.add_argument("--out", type=Path, default=ROOT / read_json(SPEC)["output"])
    args = parser.parse_args()
    {"prepare": prepare, "collect": collect, "analyze": analyze}[args.stage](args.out)


if __name__ == "__main__":
    main()
