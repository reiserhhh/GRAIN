#!/usr/bin/env python3
"""Test fixed channel/reading pairing on new bodies, without fitting or answers."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_body_relation_graph as B
from grain_research.construction import math_channel_boundary_transport as M
from grain_research.construction import math_cold_evidence as C
from grain_research.vector_mechanisms.navigation import MODEL
from scripts.run_grain_body_relation_graph_hm89 import collect, material_tokens
from scripts.run_grain_hierarchical_query_hm88 import load, sha, table, verify, write
from scripts.run_grain_query_evidence_hm86 import seal, unit

SPEC = ROOT / "configs/grain_channel_boundary_transport_hm90.json"
ARMS = (
    "cold",
    "scalar_direct",
    "scalar_harmonic",
    "direct_paired",
    "direct_average_law",
    "harmonic_paired",
    "harmonic_average_law",
    *(f"{kind}_shift_{i:02d}" for kind in ("direct", "harmonic") for i in range(1, 12)),
)
CODE = (
    "configs/grain_channel_boundary_transport_hm90.json",
    "docs/GRAIN_CHANNEL_BOUNDARY_TRANSPORT_PLAN_HM90.md",
    "grain_research/construction/math_channel_boundary_transport.py",
    "grain_research/construction/math_body_relation_graph.py",
    "grain_research/construction/math_cold_evidence.py",
    "scripts/run_grain_channel_boundary_transport_hm90.py",
    "scripts/run_grain_body_relation_graph_hm89.py",
    "scripts/run_grain_hierarchical_query_hm88.py",
    "scripts/run_grain_query_evidence_hm86.py",
    "grain_research/vector_mechanisms/prefix.py",
    "tests/test_grain_math_channel_boundary_transport.py",
)


def sentence(event, actor, behavior, frame):
    """Expose both people with fixed report/action roles and changed framing."""
    a, r = ("A", "B") if actor > 0 else ("B", "A")
    if frame == "reporter_first":
        return f"On occasion {event}, Person {r} noted that Person {a} {behavior}."
    if frame == "actor_first":
        return (
            f"On occasion {event}, Person {a} {behavior}, as documented by Person {r}."
        )
    if frame == "embedded_report":
        return f"On occasion {event}, Person {r} filed an account in which Person {a} {behavior}."
    raise ValueError("Unknown fixed frame")


def prepare(out):
    """Seal ancestry, unchanged source functions and fresh materials before calls."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve prior/failed packets")
    spec = load(SPEC)
    source = ROOT / spec["source"]
    ancestry = load(source / "00/manifest.json")
    verify(ROOT, ancestry["code"] | ancestry["source"])
    verify(source, ancestry["inputs"])
    verify(source, load(source / "01/raw_seal.json"))
    verify(source, load(source / "02/output_seal.json"))
    if (
        load(source / "verification.json")["status"]
        != "VERIFIED_FINITE_HM89_NOT_GLOBAL"
    ):
        raise ValueError("Verified source required")
    bundle = load(source / "00/bundle.json")
    if bundle["targets"] != spec["targets"]:
        raise ValueError("Source target order differs")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    system = load(ROOT / spec["source_material"])["system"]
    rows = []
    points = list(itertools.product((-1, 1), repeat=4))
    for frame, order, target, corner in itertools.product(
        spec["frames"], (0, 1), spec["targets"], points
    ):
        x, y, g, h = corner
        actions = [
            spec["behaviours"][target][int(x > 0)],
            spec["behaviours"][target][int(y > 0)],
        ]
        lines = [
            sentence("K1", g, actions[0], frame),
            sentence("K2", h, actions[1], frame),
            "On occasion K3, Person A and Person B compared a train timetable.",
        ]
        order_ids = [0, 1, 2] if order == 0 else [2, 1, 0]
        body = "\n".join(lines[i] for i in order_ids)
        rows.append(
            {
                "unit": len(rows),
                "frame": frame,
                "order": order,
                "target": target,
                "coordinates": list(corner),
                "event_order": order_ids,
                **material_tokens(body, target, system, tokenizer, actions),
            }
        )
    if len(rows) != spec["max_forwards"]:
        raise ValueError("Incomplete material")
    lookup = {(r["frame"], r["order"], r["target"], *r["coordinates"]): r for r in rows}
    bags = 0
    for r in rows:
        x, y, g, h = r["coordinates"]
        for gg, hh in ((-g, h), (g, -h)):
            other = lookup[r["frame"], r["order"], r["target"], x, y, gg, hh]
            if sorted(r["input_ids"]) != sorted(other["input_ids"]):
                raise ValueError("Role exchange changes token bag")
            bags += 1
    (out / "00").mkdir(parents=True)
    write(out / "00/rows.json", rows)
    write(out / "00/bundle.json", bundle)
    write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "arms": ARMS,
            "code": {n: sha(ROOT / n) for n in CODE},
            "source": {
                str(p.relative_to(ROOT)): sha(p)
                for p in (
                    source / "00/bundle.json",
                    source / "00/manifest.json",
                    source / "verification.json",
                    source / "02/output_seal.json",
                )
            },
            "inputs": seal(out, out / "00"),
            "model": ancestry["model"],
            "paired_token_bags": bags,
            "no_application_role_labels_in_graph": True,
            "no_application_specific_query_answer_or_fit": True,
            "generic_assessment_system_retained": True,
            "HM87_failed_prefix_gate_retained": ancestry[
                "HM87_failed_prefix_gate_retained"
            ],
        },
    )
    print("PREPARED HM90", len(rows), flush=True)


def observe(raw, row, bundle):
    """Keep full channel laws, every pairing shift and exact reading accounts."""
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
    transitions, self_mass = M.channel_graphs(raw["q_body"], k)
    scalar, _ = B.causal_graph(raw["q_body"], k)
    laws = {
        kind: M.channel_laws(transitions, labels, kind == "direct")
        for kind in ("direct", "harmonic")
    }
    collapse = M.harmonic_collapse_account(transitions, laws["harmonic"], labels)
    n, heads = len(k), len(transitions)
    gates = {
        "cold": np.ones((heads, n)),
        "scalar_direct": np.broadcast_to(
            B.direct_law(scalar, labels)[:, 0], (heads, n)
        ),
        "scalar_harmonic": np.broadcast_to(
            B.boundary_law(scalar, labels)[:, 0], (heads, n)
        ),
    }
    for kind, h in laws.items():
        gate = h[:, :, 0]
        gates[kind + "_paired"] = gate
        gates[kind + "_average_law"] = np.broadcast_to(gate.mean(0), gate.shape)
        gates.update(
            {f"{kind}_shift_{i:02d}": np.roll(gate, i, axis=0) for i in range(1, heads)}
        )
    expanded = np.repeat(v, heads // v.shape[1], axis=1)
    directions = np.array(bundle["directions"])
    values, raw_values = np.zeros((len(ARMS), 6)), np.zeros((len(ARMS), 6))
    accounts, errors = [], [collapse["identity_error"]]
    for qi in range(6):
        q = C.rotate(
            bundle["raw_queries"][qi],
            positions[-1] + bundle["gaps"][qi],
            bundle["frequencies"],
            bundle["scaling"],
        )
        logits = np.stack(
            [k[:, h // (heads // k.shape[1])] @ q[h] for h in range(heads)]
        )
        projected = np.einsum("ihd,hd->hi", expanded, directions[qi])
        obs = {name: M.measure(logits, projected, gates[name], ids) for name in ARMS}
        for ai, name in enumerate(ARMS):
            values[ai, qi], raw_values[ai, qi] = (
                obs[name]["score"],
                obs[name]["raw_action"],
            )
            errors.append(
                float(
                    np.max(
                        np.abs(
                            obs[name]["heads"] - obs[name]["event_contribution"].sum(1)
                        )
                    )
                )
            )
        for kind in laws:
            pair, mean = obs[kind + "_paired"], obs[kind + "_average_law"]
            orbit = [pair, *(obs[f"{kind}_shift_{i:02d}"] for i in range(1, heads))]
            account = M.conditioning_account(pair, mean)
            mass_error = float(
                np.max(np.abs(np.mean([o["mass"] for o in orbit], 0) - mean["mass"]))
            )
            raw_error = float(
                np.max(
                    np.abs(
                        np.mean([o["numerator"] for o in orbit], 0) - mean["numerator"]
                    )
                )
            )
            simultaneous = M.measure(
                logits[::-1], projected[::-1], gates[kind + "_paired"][::-1], ids
            )
            gauge_error = abs(simultaneous["score"] - pair["score"])
            errors.extend(
                (account["identity_error"], mass_error, raw_error, gauge_error)
            )
            accounts.append(
                {
                    "unit": row["unit"],
                    "frame": row["frame"],
                    "order": row["order"],
                    "target": row["target"],
                    "readout": bundle["targets"][qi],
                    "kind": kind,
                    **account,
                    "paired_raw_action": pair["raw_action"],
                    "mean_raw_action": mean["raw_action"],
                    "paired_mean_mass": float(pair["mass"].mean()),
                    "average_mean_mass": float(mean["mass"].mean()),
                    "orbit_conditional_remainder": float(
                        np.mean([o["score"] for o in orbit]) - mean["score"]
                    ),
                    "orbit_mass_error": mass_error,
                    "orbit_numerator_error": raw_error,
                    "synchronous_gauge_error": gauge_error,
                }
            )
    role = []
    for kind, h in laws.items():
        for event, span in enumerate(row["action_groups"]):
            signs = (h[:, span, 0] - h[:, span, 1]).mean(1)
            role.append(
                {
                    "unit": row["unit"],
                    "frame": row["frame"],
                    "order": row["order"],
                    "target": row["target"],
                    "kind": kind,
                    "event": event,
                    "mean_actor_alignment": float(
                        signs.mean() * row["coordinates"][event + 2]
                    ),
                    "head_allocation_std": float(signs.std()),
                    "unresolved": float(h[:, span, 2].mean()),
                }
            )
    graph = {
        "unit": row["unit"],
        "frame": row["frame"],
        "order": row["order"],
        "target": row["target"],
        "solving_collapse_rms": collapse["collapse_rms"],
        "solving_covariance_rms": float(np.sqrt(np.mean(collapse["covariance"] ** 2))),
        "normalizing_collapse_rms": float(
            np.sqrt(np.mean((scalar - transitions.mean(0)) ** 2))
        ),
        "solving_identity_error": collapse["identity_error"],
        "removed_self_mass": float(self_mass.mean()),
    }
    return values, raw_values, accounts, role, graph, max(errors)


def analyze(out):
    """Measure paired action and conditioning, not a best accuracy score."""
    manifest, rows, bundle = (
        load(out / "00/manifest.json"),
        load(out / "00/rows.json"),
        load(out / "00/bundle.json"),
    )
    verify(ROOT, manifest["code"] | manifest["source"])
    verify(out, manifest["inputs"])
    verify(out, load(out / "01/raw_seal.json"))
    shape = (3, 2, 6, 16, len(ARMS), 6)
    fields, raw_fields = np.zeros(shape), np.zeros(shape)
    accounts, roles, graphs, errors = [], [], [], []
    points = list(itertools.product((-1, 1), repeat=4))
    for row in rows:
        value, raw_value, account, role, graph, error = observe(
            unit(out, row["unit"]), row, bundle
        )
        index = (
            manifest["spec"]["frames"].index(row["frame"]),
            row["order"],
            bundle["targets"].index(row["target"]),
            points.index(tuple(row["coordinates"])),
        )
        fields[index], raw_fields[index] = value, raw_value
        accounts.extend(account)
        roles.extend(role)
        graphs.append(graph)
        errors.append(error)
    if max(errors) > 1e-9:
        raise ValueError("Exact numerical identity failed")
    pa, pb = B.author_laws(np.array(points), manifest["spec"]["epsilon"])
    contrasts = []
    for fi, order, ti, ai, ri in itertools.product(
        range(3), range(2), range(6), range(len(ARMS)), range(6)
    ):
        contrasts.append(
            {
                "frame": manifest["spec"]["frames"][fi],
                "order": order,
                "target": bundle["targets"][ti],
                "readout": bundle["targets"][ri],
                "arm": ARMS[ai],
                "author_conditional_contrast": float(
                    (pa - pb) @ fields[fi, order, ti, :, ai, ri]
                ),
                "author_raw_contrast": float(
                    (pa - pb) @ raw_fields[fi, order, ti, :, ai, ri]
                ),
                "is_target": ti == ri,
            }
        )
    summary = {
        "status": "EXECUTED_FINITE_HM90_AWAITING_VERIFIER",
        "calls": len(rows),
        "model_seconds": load(out / "01/status.json")["model_seconds"],
        "identity_error": max(errors),
        "mean_solving_collapse_rms": float(
            np.mean([r["solving_collapse_rms"] for r in graphs])
        ),
        "mean_normalizing_collapse_rms": float(
            np.mean([r["normalizing_collapse_rms"] for r in graphs])
        ),
        "all_heads_and_all_shifts_retained": True,
        "no_application_role_labels_or_queries_in_computation": True,
        "no_new_fitting": True,
        "encoder_and_source_calibration_still_neural": True,
        "original_scope_complete": False,
    }
    (out / "02").mkdir()
    np.savez_compressed(
        out / "02/projections.npz",
        fields=fields,
        raw_fields=raw_fields,
        pplus=pa,
        pminus=pb,
    )
    table(out / "02/conditioning_accounts.csv", accounts)
    table(out / "02/token_role_channels.csv", roles)
    table(out / "02/graph_collapse_accounts.csv", graphs)
    table(out / "02/author_law_contrasts.csv", contrasts)
    write(out / "02/summary.json", summary)
    write(out / "02/output_seal.json", seal(out, out / "02"))
    print(summary, flush=True)


def main():
    """Run a single finite stage without automatic follow-up experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "collect", "analyze"))
    parser.add_argument("--output", type=Path, default=ROOT / load(SPEC)["output"])
    args = parser.parse_args()
    {"prepare": prepare, "collect": collect, "analyze": analyze}[args.stage](
        args.output
    )


if __name__ == "__main__":
    main()
