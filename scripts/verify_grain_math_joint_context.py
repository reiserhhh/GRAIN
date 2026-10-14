#!/usr/bin/env python3
"""Alternate-formula HM-08 verification plus exact contextual-action diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from run_grain_math_binding import load_model, read_bank
from run_grain_math_response import MODELS, ROOT, sha, write_csv, write_json

from grain_research.construction import math_context_operator as C
from grain_research.construction import math_joint_context as J

DEFAULT = ROOT / "results/grain_math_response_v1/joint_context_r1"
VERIFY = ROOT / "results/grain_math_response_v1/joint_context_verification_r1"


def vec(refs, old, new):
    """Resolve cached numeric vectors without calling the experimental scorer."""
    return np.array([old[i] if s == "source" else new[i] for s, i in refs])


def affine(model, x):
    """Expand the intercept separately as a cold alternate affine formula."""
    return x @ model["w"] + (model["bias"] - model["mu"] @ model["w"])


def qualify_readout(z, events, account):
    """Independently compute role/mode projection, soft unit and hard coverage."""
    if account == "desire_about_other":
        role = np.array(
            [e["conceiver"] == "self" and e["actor"] == "other" for e in events]
        )
        active = np.array([e["mode"] == "desire" for e in events])
    else:
        role = np.array(
            [
                e["conceiver"] == "self"
                if account == "state"
                else e["actor"] == "self"
                and (
                    e["conceiver"] == "self" or e["mode"] in {"actual", "hypothetical"}
                )
                for e in events
            ]
        )
        active = np.array([e["mode"] == account for e in events])
    mass = np.clip((z[:, 0] + 1) / 2, 0, 1) * role * active
    values = mass if account == "state" else mass * z[:, 1]
    eligible = (z[:, 0] > 0) & role & active
    return values, mass, eligible


def verify(out, dest):
    """Recompute every coordinate/purpose and actual-vector context decomposition."""
    if (dest / "verification.json").exists():
        raise ValueError("verification output frozen; use a new destination")
    hashes = json.loads((out / "04/artifact_hashes.json").read_text())
    for name, digest in hashes.items():
        if sha(out / name) != digest:
            raise ValueError("changed primary: " + name)
    protocol = json.loads((out / "00/protocol_before_run.json").read_text())
    source = ROOT / protocol["spec"]["source_packet"]
    for name, digest in json.loads(
        (source / "04/artifact_hashes.json").read_text()
    ).items():
        if sha(source / name) != digest:
            raise ValueError("changed source: " + name)
    bank = read_bank(out / "00/documents.jsonl")
    indexed = json.loads((out / "01/observation_indices.json").read_text())
    event_rows = list(csv.DictReader((out / "02/event_readouts.csv").open()))
    predictions = list(csv.DictReader((out / "02/predictions.csv").open()))
    source_lookup = {
        r["text"]: i for i, r in enumerate(read_bank(source / "00/material_bank.jsonl"))
    }
    actual = defaultdict(list)
    for e in event_rows:
        actual[e["reader"], e["material_id"], e["arm"]].append(e)
    maps, errors, context_rows = {}, defaultdict(float), []
    for reader in MODELS:
        old = np.load(source / f"01B/embeddings/{reader}.npy").astype(float)
        new = np.load(out / f"01B/embeddings/{reader}.npy").astype(float)
        for target in sorted({r["target"] for r in bank}):
            model = load_model(source / f"02/{reader}_{target}.npz")
            op = C.compile_tensor(model["auto_tensor"])
            selected = [i for i, r in enumerate(bank) if r["target"] == target]
            for arm in protocol["spec"]["arms"]:
                events, slices = [], {}
                for i in selected:
                    a = len(events)
                    local = indexed[i]["declared" if arm == "declared" else "observed"]
                    key = (
                        "legal_ref"
                        if arm == "legal_joint"
                        else "naive_ref"
                        if arm == "naive_joint"
                        else "object_ref"
                    )
                    events.extend({**e, "object_ref": e[key]} for e in local)
                    slices[i] = (a, len(events))
                z = affine(
                    model["whole_factors"],
                    vec([e["readout"] for e in events], old, new),
                )
                if arm != "whole":
                    z[:, 0] = C.score_compiled(
                        op,
                        vec([e["action_ref"] for e in events], old, new),
                        vec([e["object_ref"] for e in events], old, new),
                    )
                if arm == "conservative":
                    aa, oo, candidate_slices = [], [], []
                    for e in events:
                        a = len(oo)
                        oo.extend(e["candidate_refs"])
                        aa.extend([e["action_ref"]] * len(e["candidate_refs"]))
                        candidate_slices.append((a, len(oo)))
                    candidate_scores = C.score_compiled(
                        op, vec(aa, old, new), vec(oo, old, new)
                    )
                    for j, (a, b) in enumerate(candidate_slices):
                        z[j, 0] = min(candidate_scores[a:b])
                        assert J.envelope(candidate_scores[a:b]) == J.envelope(
                            candidate_scores[a:b][::-1]
                        )
                for i in selected:
                    a, b = slices[i]
                    zz, ee = z[a:b], events[a:b]
                    saved = actual[reader, bank[i]["material_id"], arm]
                    if len(saved) != len(zz):
                        raise ValueError("coordinate row count changed")
                    given = np.array(
                        [[float(e[k]) for k in ("k", "p", "o", "a")] for e in saved]
                    ).reshape(-1, 4)
                    errors["coordinates"] = max(
                        errors["coordinates"],
                        float(np.max(np.abs(given - zz))) if len(zz) else 0,
                    )
                    maps[reader, bank[i]["material_id"], arm] = (zz, ee)
            # Exact decomposition on every actual strict -> legal/naive object change.
            observed = [(i, e) for i in selected for e in indexed[i]["observed"]]
            action = vec([e["action_ref"] for _, e in observed], old, new)
            before = vec([e["object_ref"] for _, e in observed], old, new)
            alt_text = (
                "Action: "
                + ("photograph" if target == "coordination_action" else "decorate")
                + "."
            )
            alt = np.repeat(
                old[source_lookup[alt_text]][None, :], len(observed), axis=0
            )
            for view, field in (
                ("legal_joint", "legal_ref"),
                ("naive_joint", "naive_ref"),
            ):
                after = vec([e[field] for _, e in observed], old, new)
                add, coupled, total = C.context_response(op, action, before, after)
                f00, f01 = (
                    C.score_compiled(op, alt, before),
                    C.score_compiled(op, alt, after),
                )
                f10, f11 = (
                    C.score_compiled(op, action, before),
                    C.score_compiled(op, action, after),
                )
                corner = C.four_corner(op, alt, action, before, after)
                errors["context_response"] = max(
                    errors["context_response"],
                    float(np.max(np.abs(total - (f11 - f10)))),
                )
                errors["four_corner"] = max(
                    errors["four_corner"],
                    float(
                        np.max(np.abs(corner - J.cross_difference(f00, f01, f10, f11)))
                    ),
                )
                for j, (i, e) in enumerate(observed):
                    context_rows.append(
                        {
                            "reader": reader,
                            "target": target,
                            "material_id": bank[i]["material_id"],
                            "node_id": e["node_id"],
                            "view": view,
                            "variant": bank[i]["variant"],
                            "changed": e["object_ref"] != e[field],
                            "additive_object_shift": float(add[j]),
                            "action_coupled_shift": float(coupled[j]),
                            "total_shift": float(total[j]),
                            "four_corner": float(corner[j]),
                            "alternative_action": alt_text,
                            "embedding_displacement": float(
                                np.linalg.norm(after[j] - before[j])
                            ),
                        }
                    )
    for row in predictions:
        z, events = maps[row["reader"], row["material_id"], row["arm"]]
        values, mass, eligible = qualify_readout(z, events, row["account"])
        checks = [
            (float(values.sum()), float(row["numerator"])),
            (float(mass.sum()), float(row["soft_mass"])),
        ]
        if row["complete"] == "True":
            checks.append((float(values.sum()), float(row["unit"])))
            if int(eligible.sum()) != int(row["observed_events"]):
                raise ValueError("hard qualification differs")
            if row["profile"]:
                checks.append((float(values[eligible].mean()), float(row["profile"])))
        errors["purpose_values"] = max(
            errors["purpose_values"], max(abs(a - b) for a, b in checks)
        )
    if any(v > 1e-9 for v in errors.values()):
        raise ValueError("alternate formulas disagree: " + repr(dict(errors)))
    write_csv(dest / "context_response_decomposition.csv", context_rows)
    groups = defaultdict(list)
    for r in context_rows:
        if r["changed"]:
            groups[r["reader"], r["target"], r["view"], r["variant"]].append(r)
    response_summary = [
        {
            **dict(zip(("reader", "target", "view", "variant"), key, strict=True)),
            "changed_rows": len(rows),
            **{
                f"{k}_rms": float(np.sqrt(np.mean([r[k] ** 2 for r in rows])))
                for k in (
                    "additive_object_shift",
                    "action_coupled_shift",
                    "total_shift",
                    "four_corner",
                )
            },
            "unit_warning": "correlated additive/coupled terms; RMS ratio is not an information fraction",
        }
        for key, rows in groups.items()
    ]
    write_csv(dest / "context_response_summary.csv", response_summary)
    pred_lookup = {
        (p["reader"], p["material_id"], p["arm"], p["account"]): p for p in predictions
    }
    panels = list(csv.DictReader((out / "03/author_profiles.csv").open()))
    by_material = {r["material_id"]: r for r in bank}
    single_lookup = {
        (
            r["target"],
            r["family"],
            r["object_family"],
            r["variant"],
            r["events"][0]["mode"],
            r["events"][0]["bits"][1],
            r["events"][0]["actor"],
        ): r["material_id"]
        for r in bank
        if r["kind"] == "single"
    }
    for p in panels:
        endpoints = []
        author = int(p["author_program"])
        for c in (-1, 1):
            values = []
            for j in (0, 1):
                polarity = -1 if (author + j + (c == 1)) % 3 == 0 else 1
                mid = single_lookup[
                    p["target"],
                    (author + j) % 4,
                    j,
                    "reduced_positive" if c == -1 else "finite_positive",
                    p["account"],
                    polarity,
                    1,
                ]
                r = pred_lookup[p["reader"], mid, p["arm"], p["account"]]
                if r["complete"] == "True" and int(r["observed_events"]):
                    values.append(float(r["profile"]))
                assert by_material[mid]["kind"] == "single"
            endpoints.append(float(np.mean(values)) if values else None)
        if all(x is not None for x in endpoints):
            errors["author_endpoints"] = max(
                errors["author_endpoints"],
                abs((endpoints[0] + endpoints[1]) / 2 - float(p["level"])),
                abs((endpoints[1] - endpoints[0]) / 2 - float(p["response"])),
            )
        elif p["level"] or p["response"]:
            raise ValueError("unobserved author endpoint filled as neutral")
    if errors["author_endpoints"] > 1e-9:
        raise ValueError("author aggregation differs")
    panel_groups = defaultdict(list)
    for p in panels:
        panel_groups[p["reader"], p["target"], p["account"], p["arm"]].append(p)
    author_metrics = []
    for key, rows in panel_groups.items():
        good = [r for r in rows if r["level"] and r["response"]]
        author_metrics.append(
            {
                **dict(zip(("reader", "target", "account", "arm"), key, strict=True)),
                "rows": len(rows),
                "scored": len(good),
                **{
                    f"{k}_rmse": float(
                        np.sqrt(
                            np.mean(
                                [
                                    (float(r[k]) - float(r["expected_" + k])) ** 2
                                    for r in good
                                ]
                            )
                        )
                    )
                    if good
                    else None
                    for k in ("level", "response")
                },
            }
        )
    write_csv(dest / "author_metrics.csv", author_metrics)
    result = {
        "primary": str(out),
        "verified_hashes": len(hashes),
        "coordinate_rows": len(event_rows),
        "purpose_rows": len(predictions),
        "author_program_rows": len(panels),
        "context_decomposition_rows": len(context_rows),
        "max_errors": dict(errors),
        "same_agent_alternate_formula": True,
        "human_independent_review": False,
        "new_encodings": 0,
        "new_fits": 0,
        "goal_status": "active",
    }
    write_json(dest / "verification.json", result)
    write_json(
        dest / "artifact_hashes.json",
        {
            str(p.relative_to(dest)): sha(p)
            for p in sorted(dest.rglob("*"))
            if p.is_file()
        },
    )
    print(json.dumps(result), flush=True)


def main():
    """Verify one immutable packet without changing its scientific outputs."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT)
    p.add_argument("--dest", type=Path, default=VERIFY)
    args = p.parse_args()
    verify(args.out.resolve(), args.dest.resolve())


if __name__ == "__main__":
    main()
