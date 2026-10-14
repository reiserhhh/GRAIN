#!/usr/bin/env python3
"""Run a finite preanswer history-conditioned event observation experiment."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_channel_boundary_transport as B
from grain_research.construction import math_cold_evidence as C
from grain_research.construction import math_event_history_observation as M
from grain_research.construction import math_relation_scope as R
from grain_research.vector_mechanisms.navigation import MODEL
from scripts.run_grain_body_relation_graph_hm89 import collect, material_tokens
from scripts.run_grain_hierarchical_query_hm88 import (
    load,
    sha,
    table,
    unit,
    verify,
    write,
)
from scripts.run_grain_query_evidence_hm86 import seal
from scripts.run_grain_relation_scope_hm91 import sentence

SPEC = ROOT / "configs/grain_event_history_observation_hm92.json"
FIELDS = (
    "full_cold_raw",
    "full_paired_raw",
    "K1_local_cold",
    "K1_local_paired",
    "K1_uniform_value",
    "K2_local_cold",
    "K2_local_paired",
    "K2_uniform_value",
    "K1_global_raw",
    "K2_global_raw",
)
CODE = (
    "configs/grain_event_history_observation_hm92.json",
    "docs/GRAIN_EVENT_HISTORY_OBSERVATION_PLAN_HM92.md",
    "grain_research/construction/math_event_history_observation.py",
    "scripts/run_grain_event_history_observation_hm92.py",
    "tests/test_grain_math_event_history_observation.py",
    "scripts/run_grain_relation_scope_hm91.py",
    "grain_research/construction/math_relation_scope.py",
    "grain_research/construction/math_channel_boundary_transport.py",
    "grain_research/construction/math_cold_evidence.py",
    "scripts/run_grain_body_relation_graph_hm89.py",
    "scripts/run_grain_hierarchical_query_hm88.py",
    "scripts/run_grain_query_evidence_hm86.py",
    "grain_research/vector_mechanisms/prefix.py",
)


def prepare(out):
    """Authenticate source and check all bag/position/prefix controls pre-call."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve prior or partial results")
    spec = load(SPEC)
    source = ROOT / spec["source"]
    ancestry = load(source / "00/manifest.json")
    verify(ROOT, ancestry["code"] | ancestry["source"])
    verify(source, ancestry["inputs"])
    verify(source, load(source / "01/raw_seal.json"))
    verify(source, load(source / "02/output_seal.json"))
    if (
        load(source / "verification.json")["status"]
        != "VERIFIED_FINITE_HM91_NOT_GLOBAL"
    ):
        raise ValueError("Verified HM91 source required")
    bundle = load(source / "00/bundle.json")
    if bundle["targets"] != spec["targets"]:
        raise ValueError("Source targets changed")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    system = load(ROOT / spec["source_material"])["system"]
    rows = []
    for frame, order, target, coords in itertools.product(
        spec["frames"],
        range(2),
        spec["targets"],
        itertools.product((-1, 1), repeat=6),
    ):
        x, y, g, h, a, b = coords
        actions = [
            spec["behaviours"][target][int(x > 0)],
            spec["behaviours"][target][int(y > 0)],
        ]
        lines = [
            sentence(
                "K1",
                g,
                actions[0],
                frame,
                "negated_event" if a > 0 else "negated_report",
            ),
            sentence(
                "K2",
                h,
                actions[1],
                frame,
                "negated_event" if b > 0 else "negated_report",
            ),
            "On occasion K3, Person A and Person B compared the departure times for a train.",
        ]
        permutation = spec["orders"][order]
        body = "\n".join(lines[i] for i in permutation)
        rows.append(
            {
                "unit": len(rows),
                "frame": frame,
                "order": order,
                "target": target,
                "coordinates": list(coords),
                "event_order": permutation,
                **material_tokens(body, target, system, tokenizer, actions),
            }
        )
    if len(rows) != spec["max_forwards"]:
        raise ValueError("Incomplete full design")
    lookup = {(r["frame"], r["order"], r["target"], *r["coordinates"]): r for r in rows}
    controls = {
        "scope_bags": 0,
        "actor_bags": 0,
        "K2_action_positions": 0,
        "future_prefix_inputs": 0,
    }
    for r in rows:
        coords = r["coordinates"]
        for axis in (2, 3, 4, 5):
            changed = list(coords)
            changed[axis] *= -1
            other = lookup[r["frame"], r["order"], r["target"], *changed]
            if sorted(r["input_ids"]) != sorted(other["input_ids"]):
                raise ValueError("Bag/length control failed;do not rescue")
            controls["scope_bags" if axis > 3 else "actor_bags"] += 1
            if axis == 4:
                if r["action_groups"][1] != other["action_groups"][1]:
                    raise ValueError("K2 positions changed")
                ix = [r["body_indices"][i] for i in r["action_groups"][1]]
                iy = [other["body_indices"][i] for i in other["action_groups"][1]]
                if ix != iy:
                    raise ValueError("K2 absolute action positions changed")
                if [r["input_ids"][i] for i in ix] != [
                    other["input_ids"][i] for i in iy
                ]:
                    raise ValueError("K2 action token sequence changed")
                controls["K2_action_positions"] += 1
            early = 1 if r["order"] == 1 else 0
            if axis == (4 if early == 1 else 5):
                stop = max(r["body_indices"][i] for i in r["action_groups"][early]) + 1
                if r["input_ids"][:stop] != other["input_ids"][:stop]:
                    raise ValueError("Future flip changed early prefix")
                controls["future_prefix_inputs"] += 1
    (out / "00").mkdir(parents=True)
    write(out / "00/rows.json", rows)
    write(out / "00/bundle.json", bundle)
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
                    source / "00/manifest.json",
                    source / "verification.json",
                    source / "02/output_seal.json",
                )
            },
            "inputs": seal(out, out / "00"),
            "model": ancestry["model"],
            "precall_controls": controls,
            "no_application_question_answer_fit": True,
            "HM87_failed_prefix_gate_retained": ancestry[
                "HM87_failed_prefix_gate_retained"
            ],
        },
    )
    print("PREPARED HM92", len(rows), controls, flush=True)


def observe(raw, row, bundle):
    """Read global and span-local units with all query/gate/value terms retained."""
    ix = row["body_indices"]
    k, v = raw["k"][ix], raw["v"][ix]
    transition, _ = B.channel_graphs(raw["q_body"], k)
    gate = B.channel_laws(transition, row["root_labels"], direct=True)[:, :, 0]
    heads = len(transition)
    expanded = np.repeat(v, heads // v.shape[1], axis=1)
    groups = [np.asarray(g, int) for g in row["action_groups"]]
    event_groups = [
        np.asarray(
            [
                j
                for j, p in enumerate(ix)
                if p in row["groups"][row["event_order"].index(e)]
            ],
            int,
        )
        for e in range(2)
    ]
    fields = np.zeros((len(FIELDS), 6))
    factors = []
    for qi, direction in enumerate(bundle["directions"]):
        q = C.rotate(
            bundle["raw_queries"][qi],
            ix[-1] + bundle["gaps"][qi],
            bundle["frequencies"],
            bundle["scaling"],
        )
        logits = np.stack(
            [k[:, h // (heads // k.shape[1])] @ q[h] for h in range(heads)]
        )
        p = softmax(logits, axis=1)
        c = np.einsum("ihd,hd->hi", expanded, direction)
        fields[:2, qi] = [np.sum(p * c), np.sum(p * gate * c)]
        by_event = []
        for event, group in enumerate(groups):
            local = softmax(logits[:, group], axis=1)
            g, value = gate[:, group], c[:, group]
            fields[2 + 3 * event : 5 + 3 * event, qi] = [
                np.sum(local * value),
                np.sum(local * g * value),
                np.sum(value.mean(1)),
            ]
            eg = event_groups[event]
            fields[8 + event, qi] = np.sum(p[:, eg] * gate[:, eg] * c[:, eg])
            by_event.append((local, g, value))
        factors.append(by_event)
    means = np.stack([v[g].mean(0).ravel() for g in groups])
    return fields, factors, means


def analyze(out):
    """Resolve directed history response and equal-event-marginal author laws."""
    manifest, rows, bundle = (
        load(out / "00/manifest.json"),
        load(out / "00/rows.json"),
        load(out / "00/bundle.json"),
    )
    verify(ROOT, manifest["code"] | manifest["source"])
    verify(out, manifest["inputs"])
    verify(out, load(out / "01/raw_seal.json"))
    if load(out / "01/status.json")["state"] != "COLLECTED":
        raise ValueError("Collection incomplete")
    dest = out / "02"
    if dest.exists():
        raise FileExistsError("Preserve old analyses or failures")
    dest.mkdir()
    spec = manifest["spec"]
    points, basis, names, _ = M.design_basis()
    fields = np.zeros((2, 2, 6, 64, len(FIELDS), 6))
    raw_means = None
    previous, accounts, controls = {}, [], []
    for row in rows:
        key = (
            spec["frames"].index(row["frame"]),
            row["order"],
            spec["targets"].index(row["target"]),
            points.tolist().index(row["coordinates"]),
        )
        value, factors, means = observe(unit(out, row["unit"]), row, bundle)
        fields[key] = value
        if raw_means is None:
            raw_means = np.zeros((2, 2, 6, 64, 2, means.shape[-1]))
        raw_means[key] = means
        x, y, g, h, a, b = row["coordinates"]
        base = (key[0], key[1], key[2], x, y, g, h, b)
        if a < 0:
            previous[base] = factors
        else:
            for qi, target in enumerate(spec["targets"]):
                account = R.product_account(previous[base][qi][1], factors[qi][1])
                accounts.append(
                    {
                        "unit": row["unit"],
                        "frame": row["frame"],
                        "order": row["order"],
                        "family": row["target"],
                        "readout": target,
                        **account,
                    }
                )
    coef = np.einsum("kc,fotcaq->fotkaq", basis / 64, fields)
    plus, minus = M.author_laws(spec["epsilon"])
    laws, composition, readers, coefficient_rows = [], [], [], []
    for fi, frame in enumerate(spec["frames"]):
        for order in range(2):
            for ti, target in enumerate(spec["targets"]):
                for field, field_name in enumerate(FIELDS):
                    account = M.composition_account(fields[fi, order, ti, :, field])
                    composition.append(
                        {
                            "frame": frame,
                            "order": order,
                            "family": target,
                            "field": field_name,
                            **account,
                        }
                    )
                    for qi, readout in enumerate(spec["targets"]):
                        contrast = np.sum(
                            (plus - minus) * fields[fi, order, ti, :, field, qi]
                        )
                        laws.append(
                            {
                                "frame": frame,
                                "order": order,
                                "family": target,
                                "field": field_name,
                                "readout": readout,
                                "contrast": contrast,
                                "coefficient_formula": 2
                                * spec["epsilon"]
                                * coef[fi, order, ti, names.index("yha"), field, qi],
                            }
                        )
                        for ki, name in enumerate(names):
                            coefficient_rows.append(
                                {
                                    "frame": frame,
                                    "order": order,
                                    "family": target,
                                    "field": field_name,
                                    "readout": readout,
                                    "component": name,
                                    "value": coef[fi, order, ti, ki, field, qi],
                                }
                            )
                for field, field_name in enumerate(FIELDS):
                    matrix = (
                        2
                        * spec["epsilon"]
                        * coef[fi, order, :, names.index("yha"), field]
                    )
                    if ti == 0:
                        readers.append(
                            {
                                "frame": frame,
                                "order": order,
                                "field": field_name,
                                **M.reader_account(matrix),
                            }
                        )
            for ti, target in enumerate(spec["targets"]):
                for event in range(2):
                    axis = 4 if event == 1 else 5
                    values, read_values = [], []
                    for pi, point in enumerate(points):
                        if point[axis] > 0:
                            other = point.copy()
                            other[axis] *= -1
                            pj = points.tolist().index(other.tolist())
                            values.append(
                                raw_means[fi, order, ti, pi, event]
                                - raw_means[fi, order, ti, pj, event]
                            )
                            read_values.append(
                                fields[fi, order, ti, pi, 2 + 3 * event : 5 + 3 * event]
                                - fields[
                                    fi, order, ti, pj, 2 + 3 * event : 5 + 3 * event
                                ]
                            )
                    values, read_values = np.asarray(values), np.asarray(read_values)
                    future = (order == 1 and event == 1) or (order == 0 and event == 0)
                    controls.append(
                        {
                            "frame": frame,
                            "order": order,
                            "family": target,
                            "event": event + 1,
                            "changed_other_scope_is_future": future,
                            "value_max_abs": float(np.abs(values).max()),
                            "value_rms": float(np.sqrt(np.mean(values**2))),
                            "local_cold_rms": float(
                                np.sqrt(np.mean(read_values[:, 0] ** 2))
                            ),
                            "local_paired_rms": float(
                                np.sqrt(np.mean(read_values[:, 1] ** 2))
                            ),
                            "uniform_value_rms": float(
                                np.sqrt(np.mean(read_values[:, 2] ** 2))
                            ),
                        }
                    )
    np.savez_compressed(
        dest / "observations.npz",
        fields=fields,
        coefficients=coef,
        raw_value_means=raw_means,
        points=points,
        basis=basis,
    )
    table(dest / "coefficients.csv", coefficient_rows)
    table(dest / "author_laws.csv", laws)
    table(dest / "composition.csv", composition)
    table(dest / "reader_accounts.csv", readers)
    table(dest / "past_future_controls.csv", controls)
    table(dest / "K2_scope_product_accounts.csv", accounts)
    future_error = max(
        c["value_max_abs"] for c in controls if c["changed_other_scope_is_future"]
    )
    write(
        dest / "summary.json",
        {
            "status": "AWAITING_ALTERNATE_NUMERICAL_REVIEW",
            "new_body_calls": len(rows),
            "fields": FIELDS,
            "precall_controls": manifest["precall_controls"],
            "future_raw_value_max_error": future_error,
            "future_control": "PASS"
            if future_error <= spec["future_prefix_atol"]
            else "FAIL_preserved",
            "max_product_identity_error": max(a["identity_error"] for a in accounts),
            "max_author_formula_error": max(
                abs(a["contrast"] - a["coefficient_formula"]) for a in laws
            ),
            "no_application_fit_question_answer": True,
            "matched_author_rows": [r for r in laws if r["family"] == r["readout"]],
            "past_future_controls": controls,
            "original_scope_complete": False,
            "scientific_scope": "History-conditioned encoder observations;not psychological accuracy or unique causal mediator",
        },
    )
    write(dest / "output_seal.json", seal(out, dest))
    print("ANALYZED HM92", "future value error", future_error, flush=True)


def main():
    """Run one explicit finite stage; never overwrite a completed result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "collect", "analyze"))
    args = parser.parse_args()
    out = ROOT / load(SPEC)["output"]
    {"prepare": prepare, "collect": collect, "analyze": analyze}[args.stage](out)


if __name__ == "__main__":
    main()
