#!/usr/bin/env python3
"""Capture fresh relation scopes with frozen body-only observation functions."""

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
from grain_research.construction import math_relation_scope as M
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

SPEC = ROOT / "configs/grain_relation_scope_hm91.json"
FIELDS = (
    "cold_raw",
    "paired_raw",
    "paired_conditional",
    "mean_mass",
    "cold_core",
    "paired_core",
)
CODE = (
    "configs/grain_relation_scope_hm91.json",
    "docs/GRAIN_RELATION_SCOPE_PLAN_HM91.md",
    "grain_research/construction/math_relation_scope.py",
    "scripts/run_grain_relation_scope_hm91.py",
    "tests/test_grain_math_relation_scope.py",
    "grain_research/construction/math_channel_boundary_transport.py",
    "grain_research/construction/math_cold_evidence.py",
    "scripts/run_grain_body_relation_graph_hm89.py",
    "scripts/run_grain_hierarchical_query_hm88.py",
    "scripts/run_grain_query_evidence_hm86.py",
    "grain_research/vector_mechanisms/prefix.py",
)


def sentence(event, actor, action, frame, mode):
    """Place negation on different clauses without changing their token bag."""
    a, r = ("A", "B") if actor > 0 else ("B", "A")
    tense, report = (
        ("does", "report") if frame == "present_report" else ("did", "claim")
    )
    if mode == "asserted_event":
        body = f"Person {r} {tense} {report} that Person {a} {tense} {action}"
    elif mode == "negated_event":
        body = f"Person {r} {tense} {report} that Person {a} {tense} not {action}"
    elif mode == "negated_report":
        body = f"Person {r} {tense} not {report} that Person {a} {tense} {action}"
    elif frame == "present_report":
        wrappers = {
            "intention": "reports that Person {a} intends to",
            "preference": "reports that Person {a} prefers to",
            "imagined": "imagines that Person {a} would",
        }
        body = f"Person {r} " + wrappers[mode].format(a=a) + f" {action}"
    else:
        wrappers = {
            "intention": "said that Person {a} planned to",
            "preference": "said that Person {a} preferred to",
            "imagined": "pictured a scene in which Person {a} would",
        }
        body = f"Person {r} " + wrappers[mode].format(a=a) + f" {action}"
    return f"On occasion {event}, {body}."


def prepare(out):
    """Authenticate the fixed source and seal all new material before calls."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve completed and partial experiments")
    spec = load(SPEC)
    source = ROOT / spec["source"]
    ancestry = load(source / "00/manifest.json")
    verify(ROOT, ancestry["code"] | ancestry["source"])
    verify(source, ancestry["inputs"])
    verify(source, load(source / "01/raw_seal.json"))
    verify(source, load(source / "02/output_seal.json"))
    if (
        load(source / "verification.json")["status"]
        != "VERIFIED_FINITE_HM90_NOT_GLOBAL"
    ):
        raise ValueError("Verified HM90 source required")
    bundle = load(source / "00/bundle.json")
    if bundle["targets"] != spec["targets"]:
        raise ValueError("Frozen target order differs")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    system = load(ROOT / spec["source_material"])["system"]
    rows = []
    corners = list(itertools.product((-1, 1), repeat=4))
    for frame, mode, target, corner in itertools.product(
        spec["frames"], spec["modes"], spec["targets"], corners
    ):
        x, y, g, h = corner
        actions = [
            spec["behaviours"][target][int(x > 0)],
            spec["behaviours"][target][int(y > 0)],
        ]
        body = "\n".join(
            (
                sentence("K1", g, actions[0], frame, mode),
                sentence("K2", h, actions[1], frame, mode),
                "On occasion K3, Person A and Person B compared a bus timetable.",
            )
        )
        rows.append(
            {
                "unit": len(rows),
                "frame": frame,
                "mode": mode,
                "target": target,
                "coordinates": list(corner),
                "event_order": [0, 1, 2],
                **material_tokens(body, target, system, tokenizer, actions),
            }
        )
    if len(rows) != spec["max_forwards"]:
        raise ValueError("Incomplete fixed design")
    lookup = {(r["frame"], r["mode"], r["target"], *r["coordinates"]): r for r in rows}
    checks = {"actor_bags": 0, "scope_bags": 0, "matched_action_tokens": 0}
    for r in rows:
        x, y, g, h = r["coordinates"]
        for gg, hh in ((-g, h), (g, -h)):
            other = lookup[r["frame"], r["mode"], r["target"], x, y, gg, hh]
            if sorted(r["input_ids"]) != sorted(other["input_ids"]):
                raise ValueError("Actor swap changed token bag")
            checks["actor_bags"] += 1
        base = lookup[r["frame"], "asserted_event", r["target"], x, y, g, h]
        for a, b in zip(r["action_groups"], base["action_groups"], strict=True):
            ids = [r["input_ids"][r["body_indices"][i]] for i in a]
            ids0 = [base["input_ids"][base["body_indices"][i]] for i in b]
            if ids != ids0:
                raise ValueError("Common behavior tokens not exactly aligned")
            checks["matched_action_tokens"] += 1
        if r["mode"] == "negated_event":
            other = lookup[r["frame"], "negated_report", r["target"], x, y, g, h]
            if sorted(r["input_ids"]) != sorted(other["input_ids"]):
                raise ValueError("Negation scopes changed complete token bag")
            checks["scope_bags"] += 1
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
            "precall_controls": checks,
            "HM87_failed_prefix_gate_retained": ancestry[
                "HM87_failed_prefix_gate_retained"
            ],
            "no_application_query_answer_fit_or_mode_oracle": True,
        },
    )
    print("PREPARED HM91", len(rows), checks, flush=True)


def observe(raw, row, bundle):
    """Apply the unchanged paired one-step kernel and retain all token accounts."""
    ix = row["body_indices"]
    k, v = raw["k"][ix], raw["v"][ix]
    transitions, _ = B.channel_graphs(raw["q_body"], k)
    gate = B.channel_laws(transitions, row["root_labels"], direct=True)[:, :, 0]
    heads = len(transitions)
    expanded = np.repeat(v, heads // v.shape[1], axis=1)
    core = np.asarray([j for g in row["action_groups"] for j in g], int)
    fields, factors = np.zeros((len(FIELDS), 6)), []
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
        mass = (p * gate).sum(1)
        if np.any(mass <= 0):
            raise ValueError("Subject observation absent; do not invent zero")
        num = (p * gate * c).sum(1)
        fields[:, qi] = [
            np.sum(p * c),
            num.sum(),
            np.sum(num / mass),
            mass.mean(),
            np.sum(p[:, core] * c[:, core]),
            np.sum(p[:, core] * gate[:, core] * c[:, core]),
        ]
        factors.append((p[:, core], gate[:, core], c[:, core]))
    return fields, factors


def analyze(out):
    """Resolve relation actions and equal-marginal composition, not accuracy."""
    manifest, rows, bundle = (
        load(out / "00/manifest.json"),
        load(out / "00/rows.json"),
        load(out / "00/bundle.json"),
    )
    verify(ROOT, manifest["code"] | manifest["source"])
    verify(out, manifest["inputs"])
    verify(out, load(out / "01/raw_seal.json"))
    if load(out / "01/status.json")["state"] != "COLLECTED":
        raise ValueError("Capture still incomplete")
    dest = out / "02"
    if dest.exists():
        raise FileExistsError("Preserve analysis failures/results")
    dest.mkdir()
    spec = manifest["spec"]
    frames, modes, targets = spec["frames"], spec["modes"], spec["targets"]
    points, basis, names = M.design_basis()
    fields = np.zeros((2, 6, 6, 16, len(FIELDS), 6))
    factors, accounts = {}, []
    error = 0.0
    for r in rows:
        key = (
            frames.index(r["frame"]),
            modes.index(r["mode"]),
            targets.index(r["target"]),
            points.tolist().index(r["coordinates"]),
        )
        value, fact = observe(unit(out, r["unit"]), r, bundle)
        fields[key] = value
        basekey = (key[0], key[2], key[3])
        if key[1] == 0:
            factors[basekey] = fact
        else:
            for qi in range(6):
                account = M.product_account(factors[basekey][qi], fact[qi])
                error = max(error, account["identity_error"])
                accounts.append(
                    {
                        "unit": r["unit"],
                        "frame": r["frame"],
                        "mode": r["mode"],
                        "target": r["target"],
                        "readout": targets[qi],
                        **account,
                    }
                )
    coeff = np.einsum("kc,fmtcaq->fmtkaq", basis / 16, fields)
    np.savez_compressed(
        dest / "observations.npz",
        fields=fields,
        coefficients=coeff,
        points=points,
        basis=basis,
    )
    coefficients_rows, geometry, scopes, authors = [], [], [], []
    plus, minus = M.relation_laws(spec["relation_weights"], spec["epsilon"])
    for fi, frame in enumerate(frames):
        for mi, mode in enumerate(modes):
            for ti, target in enumerate(targets):
                for ki, name in enumerate(names):
                    for ai, field in enumerate(FIELDS):
                        for qi, readout in enumerate(targets):
                            coefficients_rows.append(
                                {
                                    "frame": frame,
                                    "mode": mode,
                                    "target": target,
                                    "component": name,
                                    "field": field,
                                    "readout": readout,
                                    "value": coeff[fi, mi, ti, ki, ai, qi],
                                }
                            )
            for ai, field in enumerate(FIELDS):
                account = M.scalar_account(
                    coeff[fi, 0, :, 1:, ai, :], coeff[fi, mi, :, 1:, ai, :]
                )
                geometry.append(
                    {"frame": frame, "mode": mode, "field": field, **account}
                )
        for ti, target in enumerate(targets):
            for ai, field in enumerate(FIELDS):
                delta = fields[fi, 1, ti, :, ai] - fields[fi, 2, ti, :, ai]
                for qi, readout in enumerate(targets):
                    scopes.append(
                        {
                            "frame": frame,
                            "target": target,
                            "field": field,
                            "readout": readout,
                            "inner_minus_outer_mean": delta[:, qi].mean(),
                            "scope_rms": np.sqrt(np.mean(delta[:, qi] ** 2)),
                            "binding_scope_difference": np.mean(
                                (
                                    points[:, 0] * points[:, 2]
                                    + points[:, 1] * points[:, 3]
                                )
                                * delta[:, qi]
                            )
                            / 2,
                        }
                    )
                    surface = fields[fi, :, ti, :, ai, qi]
                    contrast = np.sum((plus - minus) * surface)
                    blind = np.broadcast_to(surface.mean(0), surface.shape)
                    authors.append(
                        {
                            "frame": frame,
                            "target": target,
                            "field": field,
                            "readout": readout,
                            "joint_law_contrast": contrast,
                            "mode_blind_contrast": np.sum((plus - minus) * blind),
                            "scope_formula": spec["epsilon"]
                            * np.mean(
                                (
                                    points[:, 0] * points[:, 2]
                                    + points[:, 1] * points[:, 3]
                                )
                                * delta[:, qi]
                            )
                            / 6,
                        }
                    )
    table(dest / "coefficients.csv", coefficients_rows)
    table(dest / "scalar_action.csv", geometry)
    table(dest / "scope_contrasts.csv", scopes)
    table(dest / "author_laws.csv", authors)
    table(dest / "core_product_accounts.csv", accounts)
    write(
        dest / "summary.json",
        {
            "status": "AWAITING_ALTERNATE_NUMERICAL_REVIEW",
            "new_body_calls": len(rows),
            "no_fits": True,
            "fields": FIELDS,
            "scope_token_bag_checks": manifest["precall_controls"]["scope_bags"],
            "max_product_account_error": error,
            "author_mode_marginal_error": float(
                np.max(np.abs(plus.sum(1) - minus.sum(1)))
            ),
            "author_corner_marginal_error": float(
                np.max(np.abs(plus.sum(0) - minus.sum(0)))
            ),
            "scalar_action": geometry,
            "matched_author_law_rows": [
                r
                for r in authors
                if r["target"] == r["readout"]
                and r["field"] in ("cold_raw", "paired_raw", "paired_core")
            ],
            "scientific_scope": "Frozen neural body observer;relation-composition effects,not personality accuracy or a factuality classifier",
            "original_scope_complete": False,
        },
    )
    write(dest / "output_seal.json", seal(out, dest))
    print("ANALYZED HM91", "product error", error, flush=True)


def main():
    """Run one explicit stage without overwriting prior packets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "collect", "analyze"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = args.output or ROOT / load(SPEC)["output"]
    {"prepare": prepare, "collect": collect, "analyze": analyze}[args.stage](out)


if __name__ == "__main__":
    main()
