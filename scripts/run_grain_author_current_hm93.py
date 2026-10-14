#!/usr/bin/env python3
"""Collect a frozen vector kernel and cold-score directed toy author ledgers."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_author_current as M
from grain_research.construction import math_channel_boundary_transport as B
from grain_research.construction import math_cold_evidence as C
from grain_research.vector_mechanisms.navigation import MODEL
from scripts.run_grain_body_relation_graph_hm89 import collect as collect_prefix
from scripts.run_grain_body_relation_graph_hm89 import material_tokens
from scripts.run_grain_hierarchical_query_hm88 import (
    load,
    sha,
    table,
    unit,
    verify,
    write,
)
from scripts.run_grain_query_evidence_hm86 import seal

SPEC = ROOT / "configs/grain_author_current_hm93.json"
FIELDS = ("uniform_value", "local_query", "local_bound_query")
CODE = (
    "configs/grain_author_current_hm93.json",
    "docs/GRAIN_AUTHOR_CURRENT_PLAN_HM93.md",
    "grain_research/construction/math_author_current.py",
    "scripts/run_grain_author_current_hm93.py",
    "tests/test_grain_math_author_current.py",
    "scripts/verify_grain_author_current_hm93.py",
    "scripts/run_grain_body_relation_graph_hm89.py",
    "scripts/run_grain_hierarchical_query_hm88.py",
    "scripts/run_grain_query_evidence_hm86.py",
    "scripts/run_grain_math_span_binding.py",
    "scripts/run_grain_math_construct_routing.py",
    "scripts/run_grain_upstream_routes.py",
    "grain_research/construction/math_channel_boundary_transport.py",
    "grain_research/construction/math_body_relation_graph.py",
    "grain_research/construction/math_cold_evidence.py",
    "grain_research/vector_mechanisms/navigation.py",
    "grain_research/vector_mechanisms/prefix.py",
)


def collect(out):
    """Use the existing bounded, resumable, no-answer prefix collector."""
    collect_prefix(out)


def sentence(event, action, frame):
    """State an action by the same person; framing is a declared protocol."""
    if frame == "recorded":
        return f"On occasion {event}, the written record says that Person A {action}."
    return f"On occasion {event}, Person A {action}."


def prepare(out):
    """Seal all fresh words, numerical source, paths and protocol before calls."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve earlier or partial experiments")
    spec = load(SPEC)
    source = ROOT / spec["source"]
    ancestry = load(source / "00/manifest.json")
    verify(ROOT, ancestry["code"] | ancestry["source"])
    verify(source, ancestry["inputs"])
    verify(source, load(source / "01/raw_seal.json"))
    verify(source, load(source / "02/output_seal.json"))
    if (
        load(source / "verification.json")["status"]
        != "VERIFIED_FINITE_HM92_NOT_GLOBAL"
    ):
        raise ValueError("HM92 final verification required")
    bundle = load(source / "00/bundle.json")
    if bundle["targets"] != spec["targets"]:
        raise ValueError("Target source changed")
    system = load(ROOT / spec["source_material"])["system"]
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    rows = []
    for lex, frame, family in itertools.product(
        range(2), spec["frames"], spec["targets"]
    ):
        words = spec["behaviours"][family][lex]
        for kind in ("pair", "reset"):
            for i, j in itertools.product(
                range(3) if kind == "pair" else [-1], range(3)
            ):
                previous = (
                    words[i]
                    if kind == "pair"
                    else "looked up departure times for a tram"
                )
                actions = [previous, words[j]]
                body = "\n".join(
                    [
                        sentence("K1", previous, frame),
                        sentence("K2", words[j], frame),
                        "On occasion K3, Person A checked the date on a wall calendar.",
                    ]
                )
                tokens = material_tokens(body, family, system, tokenizer, actions)
                if not all(tokens["action_groups"]):
                    raise ValueError("Empty design action span")
                rows.append(
                    {
                        "unit": len(rows),
                        "kind": kind,
                        "lexicon": lex,
                        "frame": frame,
                        "family": family,
                        "previous": i,
                        "current": j,
                        **tokens,
                    }
                )
    if len(rows) != spec["max_forwards"]:
        raise ValueError("Incomplete material battery")
    paths, truth = [], []
    for direction, seed, replicate in itertools.product(
        (-1, 0, 1), spec["seeds"], range(spec["profiles_per_direction_seed"])
    ):
        path = M.euler_path(M.cycle_counts(direction), seed * 100 + replicate)
        author = f"toy_{len(paths):03d}"
        paths.append({"author": author, "states": path[:-1].tolist()})
        truth.append(
            {
                "author": author,
                "direction": direction,
                "seed": seed,
                "replicate": replicate,
                "counts": M.cycle_counts(direction).tolist(),
            }
        )
    (out / "00").mkdir(parents=True)
    write(out / "00/rows.json", rows)
    write(out / "00/bundle.json", bundle)
    write(out / "00/application_paths.json", paths)
    write(out / "00/evaluation_truth.json", truth)
    write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "fields": FIELDS,
            "code": {n: sha(ROOT / n) for n in CODE},
            "source": {
                str(p.relative_to(ROOT)): sha(p)
                for p in (
                    source / "00/bundle.json",
                    source / "verification.json",
                    source / "00/manifest.json",
                    source / "02/output_seal.json",
                )
            },
            "inputs": seal(out, out / "00"),
            "model": ancestry["model"],
            "no_application_question_answer_fit": True,
            "query_registration": "current_action_endpoint_plus_frozen_source_gap",
            "HM87_failed_prefix_gate_retained": ancestry[
                "HM87_failed_prefix_gate_retained"
            ],
            "application_has_no_truth_fields": True,
        },
    )
    print("PREPARED HM93", len(rows), "bodies", len(paths), "toy paths", flush=True)


def observe(raw, row, bundle):
    """Read only the current action, with query registration anchored there."""
    ix = row["body_indices"]
    k, v = raw["k"][ix], raw["v"][ix]
    transition, _ = B.channel_graphs(raw["q_body"], k)
    gate = B.channel_laws(transition, row["root_labels"], direct=True)[:, :, 0]
    group = np.asarray(row["action_groups"][1], int)
    heads = len(transition)
    values = np.repeat(v, heads // v.shape[1], axis=1)
    endpoint = max(ix[i] for i in group)
    fields = np.zeros((len(FIELDS), 6))
    for qi, direction in enumerate(bundle["directions"]):
        q = C.rotate(
            bundle["raw_queries"][qi],
            endpoint + bundle["gaps"][qi],
            bundle["frequencies"],
            bundle["scaling"],
        )
        logits = np.stack(
            [k[group, h // (heads // k.shape[1])] @ q[h] for h in range(heads)]
        )
        p = softmax(logits, axis=1)
        c = np.einsum("ihd,hd->hi", values[group], direction)
        fields[:, qi] = [c.mean(1).sum(), np.sum(p * c), np.sum(p * gate[:, group] * c)]
    return fields, v[group].mean(0).ravel()


def analyze(out):
    """Keep actual transition kernels and their complete directed accounts."""
    manifest, rows, bundle = (
        load(out / "00/manifest.json"),
        load(out / "00/rows.json"),
        load(out / "00/bundle.json"),
    )
    verify(ROOT, manifest["code"] | manifest["source"])
    verify(out, manifest["inputs"])
    verify(out, load(out / "01/raw_seal.json"))
    if load(out / "01/status.json")["state"] != "COLLECTED":
        raise ValueError("Incomplete real captures")
    dest = out / "02"
    dest.mkdir()
    spec = manifest["spec"]
    kernel = np.zeros((2, 2, 6, 3, 3, 3, 6))
    resets = np.zeros((2, 2, 6, 3, 3, 6))
    raw_values, raw_reset = None, None
    for row in rows:
        field, carrier = observe(unit(out, row["unit"]), row, bundle)
        key = (
            row["lexicon"],
            spec["frames"].index(row["frame"]),
            spec["targets"].index(row["family"]),
        )
        if raw_values is None:
            raw_values = np.zeros((2, 2, 6, 3, 3, len(carrier)))
            raw_reset = np.zeros((2, 2, 6, 3, len(carrier)))
        if row["kind"] == "pair":
            kernel[key + (row["previous"], row["current"])] = field
            raw_values[key + (row["previous"], row["current"])] = carrier
        else:
            resets[key + (row["current"],)] = field
            raw_reset[key + (row["current"],)] = carrier
    np.savez_compressed(
        dest / "kernels.npz",
        kernel=kernel,
        reset=resets,
        raw_value=raw_values,
        raw_reset=raw_reset,
    )
    records = []
    p, m, null = (M.cycle_counts(d) / 45 for d in (1, -1, 0))
    for lex, frame, family, field, qi in itertools.product(
        range(2), range(2), range(6), range(3), range(6)
    ):
        f = kernel[lex, frame, family, :, :, field, qi]
        reset = np.broadcast_to(resets[lex, frame, family, :, field, qi][None], (3, 3))
        even, odd = M.kernel_parts(f)
        grad, cycle = M.hodge_current(odd)
        plus, minus, neutral = (M.readout_account(j, f) for j in (p, m, null))
        records.append(
            {
                "lexicon": spec["lexicons"][lex],
                "frame": spec["frames"][frame],
                "family": spec["targets"][family],
                "field": FIELDS[field],
                "readout": spec["targets"][qi],
                "observer_odd_rms": float(np.sqrt(np.mean(odd**2))),
                "observer_cycle_rms": float(np.sqrt(np.mean(cycle**2))),
                "observer_gradient_rms": float(np.sqrt(np.mean(grad**2))),
                "author_contrast": float(plus["value"] - minus["value"]),
                "cycle_contrast": float(plus["cycle"] - minus["cycle"]),
                "traffic_contrast": float(plus["traffic"] - minus["traffic"]),
                "reversible_author_cycle": float(neutral["cycle"]),
                "reset_contrast": float(M.contract(p - m, reset)),
                "account_error": max(plus["identity_error"], minus["identity_error"]),
                "even_odd_orthogonality": float(np.sum(even * odd)),
            }
        )
    table(dest / "kernel_accounts.csv", records)
    raw_matrix = np.einsum("ij,lfcijd->lfcd", p - m, raw_values)
    write(
        dest / "summary.json",
        {
            "status": "COLLECTED_ANALYZED_AWAITING_COLD_AND_VERIFICATION",
            "actual_forwards": len(rows),
            "model_seconds": load(out / "01/status.json")["model_seconds"],
            "raw_cycle_contrast_rms_min_max": [
                float(np.sqrt(np.mean(raw_matrix**2, axis=-1)).min()),
                float(np.sqrt(np.mean(raw_matrix**2, axis=-1)).max()),
            ],
            "all_accounts_retained": len(records),
            "no_model_selection_or_accuracy": True,
        },
    )
    write(dest / "output_seal.json", seal(out, dest))
    print("ANALYZED HM93", len(records), "complete directed accounts", flush=True)


def apply(out):
    """Cold numeric scoring: read frozen kernels and paths, never simulation truth."""
    manifest = load(out / "00/manifest.json")
    verify(out, load(out / "02/output_seal.json"))
    paths = load(out / "00/application_paths.json")
    if any(set(p) != {"author", "states"} for p in paths):
        raise ValueError("Application ledger contains truth or unknown metadata")
    with np.load(out / "02/kernels.npz", allow_pickle=False) as data:
        kernel, reset = data["kernel"].copy(), data["reset"].copy()
    spec = manifest["spec"]
    records, flows = [], []
    for path in paths:
        states = np.asarray(path["states"], int)
        for mode in ("toy_ring", "open_path"):
            j = M.path_joint(states, closed=mode == "toy_ring")
            _, a, imbalance = M.flow_parts(j)
            grad, cycle = M.hodge_current(a)
            flows.append(
                {
                    "author": path["author"],
                    "mode": mode,
                    "occupancy": (
                        np.bincount(states, minlength=3) / len(states)
                    ).tolist(),
                    "joint": j.tolist(),
                    "current": a.tolist(),
                    "gradient": grad.tolist(),
                    "cycle": cycle.tolist(),
                    "imbalance": imbalance.tolist(),
                }
            )
            for lex, frame, family in itertools.product(range(2), range(2), range(6)):
                f = kernel[lex, frame, family]
                account = M.readout_account(j, f)
                reset_kernel = np.broadcast_to(reset[lex, frame, family][None], f.shape)
                reset_odd = M.kernel_parts(reset_kernel)[1]
                reset_action = M.contract(a, reset_odd)
                for field, qi in itertools.product(range(3), range(6)):
                    records.append(
                        {
                            "author": path["author"],
                            "mode": mode,
                            "lexicon": spec["lexicons"][lex],
                            "frame": spec["frames"][frame],
                            "family": spec["targets"][family],
                            "field": FIELDS[field],
                            "readout": spec["targets"][qi],
                            **{
                                k: float(account[k][field, qi])
                                for k in ("value", "traffic", "endpoint", "cycle")
                            },
                            "reset_directional_action": float(reset_action[field, qi]),
                            "account_error": account["identity_error"],
                        }
                    )
    dest = out / "03"
    dest.mkdir()
    write(dest / "author_flows.json", flows)
    table(dest / "author_scores.csv", records)
    allowed = [
        "00/manifest.json",
        "00/application_paths.json",
        "02/output_seal.json",
        "02/kernels.npz",
    ]
    write(
        dest / "access_scope.json",
        {
            "research_inputs": allowed,
            "truth_or_text_opened_by_application": False,
            "numeric_kernel_still_requires_encoder_for_new_raw_text": True,
        },
    )
    write(dest / "output_seal.json", seal(out, dest))
    print("COLD APPLIED HM93", len(flows), "flows", len(records), "scores", flush=True)


def main():
    """Expose separately resumable, nonoverwriting finite stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "collect", "analyze", "apply"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = args.output or ROOT / load(SPEC)["output"]
    globals()[args.stage](out)


if __name__ == "__main__":
    main()
