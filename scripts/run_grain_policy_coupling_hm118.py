#!/usr/bin/env python3
"""Construct policy-indexed author/response observations without new model calls."""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_policy_coupling as M
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts import run_grain_observer_focus_hm117 as R

SPEC = ROOT / "configs/grain_policy_coupling_hm118.json"
CODE = (
    "configs/grain_policy_coupling_hm118.json",
    "docs/GRAIN_POLICY_COUPLING_PLAN_HM118.md",
    "grain_research/construction/math_policy_coupling.py",
    "scripts/run_grain_policy_coupling_hm118.py",
    "scripts/score_grain_policy_coupling_cold_hm118.py",
    "tests/test_grain_math_policy_coupling.py",
)


def save(path, **arrays):
    """Preserve completed/failed numeric exports by writing only new paths."""
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def authenticate(source):
    """Require complete actual HM117 observations, not file-existence acceptance."""
    old = R.check(source)
    for seal in ("01/raw_seal.json", "02/output_seal.json"):
        H.verify_hashes(source, H.load(source / seal))
    status, verification = (
        H.load(source / "status.json"),
        H.load(source / "verification.json"),
    )
    if (
        status["state"] != "COLLECTED"
        or status["completed"] != 244
        or status["pending"] is not None
        or verification["status"] != "VERIFIED_FINITE_HM117_NOT_GLOBAL"
    ):
        raise ValueError("Complete verified finite source required")
    return old


def check(out):
    """Check code, input and original source lineage without neural imports."""
    manifest = H.load(out / "manifest.json")
    H.verify_hashes(ROOT, manifest["code"])
    H.verify_hashes(out, manifest["inputs"])
    source = ROOT / manifest["spec"]["source"]
    authenticate(source)
    H.verify_hashes(source, manifest["source"])
    return manifest


def prepare(out):
    """Compile own-actor fields and fresh event counts before any cold scores."""
    start = time.monotonic()
    if out.exists():
        raise FileExistsError("Preserve prior or failed packet;no automatic retry")
    spec = H.load(SPEC)
    if spec["new_forwards"] or spec["fits"] or tuple(spec["policies"]) != M.POLICIES:
        raise ValueError("No inference/fitting or policy selection permitted")
    source = ROOT / spec["source"]
    old = authenticate(source)
    if spec["targets"] != old["spec"]["targets"] or spec["units"] != list(R.R.UNITS):
        raise ValueError("Keep all registered targets/units")
    with np.load(source / "02/accounts.npz", allow_pickle=False) as stream:
        raw = stream["raw"]
    if raw.shape != (5, 2, 6, 2, 2, 2, 2, 6, 4) or not np.isfinite(raw).all():
        raise ValueError("Complete native law required")
    kernels = np.empty((2, 5, 24, 2, 6, 4))
    for owner, event, focus in itertools.product(range(2), range(24), range(2)):
        f, c, b = M.ATOMS[event]
        value = raw[:, focus, f, (c + 1) // 2, (b + 1) // 2, owner, owner]
        kernels[owner, 0, event, focus] = value[0]
        kernels[owner, 1:, event, focus] = value[0] - value[1:]
    records, counts, owners = [], [], []
    for owner, recipe, seed in itertools.product(range(2), range(6), spec["seeds"]):
        n = 2 * np.random.default_rng(seed).multinomial(
            spec["base_events"], M.recipes()[recipe]
        )
        records.append(
            {
                "history": len(records),
                "slot": "A" if owner else "B",
                "recipe": recipe,
                "seed": seed,
            }
        )
        counts.append(M.route_counts(n, spec["coupling_divisor"]))
        owners.append(owner)
    inputs = out / "00"
    inputs.mkdir(parents=True)
    save(inputs / "kernels.npz", kernels=kernels, atoms=M.ATOMS)
    save(inputs / "source_recipes.npz", laws=M.recipes())
    for origin in ("instrument", "author"):
        save(
            inputs / f"ledgers_{origin}.npz",
            joint_counts=np.stack(counts),
            evidence_owner=np.array(owners),
            query_owner=np.array(owners),
            origins=np.full(len(owners), origin),
        )
    H.write(inputs / "records.json", records)
    elapsed = time.monotonic() - start
    if elapsed > spec["max_cpu_seconds"]:
        raise TimeoutError("Finite CPU preparation budget exceeded")
    H.write(
        out / "manifest.json",
        {
            "spec": spec,
            "code": {p: H.sha(ROOT / p) for p in CODE},
            "source": {
                p: H.sha(source / p)
                for p in (
                    "00/manifest.json",
                    "02/accounts.npz",
                    "02/output_seal.json",
                    "verification.json",
                    "status.json",
                )
            },
            "inputs": H.seals(out, inputs),
            "prepare_seconds": elapsed,
            "scientific_scope": "36 new numeric ledgers,not36 human authors;source endpoints reused,not new model readings;author-origin copy is an interface control only",
        },
    )
    print("PREPARED HM118", kernels.shape, len(records), elapsed, flush=True)


def cold(out):
    """Execute two isolated numeric scorers, never loading a full language model."""
    spec = check(out)["spec"]
    dest = out / "01"
    dest.mkdir()
    start = time.monotonic()
    logs = []
    for origin in ("instrument", "author"):
        command = [
            sys.executable,
            str(ROOT / CODE[4]),
            "--kernels",
            str(out / "00/kernels.npz"),
            "--ledgers",
            str(out / f"00/ledgers_{origin}.npz"),
            "--output",
            str(dest / f"{origin}_scores.npz"),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(1, spec["max_cpu_seconds"] - (time.monotonic() - start)),
            check=True,
        )
        logs.append(
            {
                "origin": origin,
                "command": command,
                "exit": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
    H.write(
        dest / "cold_execution.json",
        {"processes": logs, "seconds": time.monotonic() - start, "new_forwards": 0},
    )
    H.write(out / "01/output_seal.json", H.seals(out, dest))
    print("COLD HM118 complete", time.monotonic() - start, flush=True)


def read_scores(out):
    """Authenticate and load all numeric policy accounts."""
    H.verify_hashes(out, H.load(out / "01/output_seal.json"))
    with np.load(out / "01/instrument_scores.npz", allow_pickle=False) as stream:
        return {p: stream[p] for p in stream.files}


def analyze(out):
    """Report every target/policy/unit and keep native and action profiles separate."""
    start = time.monotonic()
    manifest, values = check(out), read_scores(out)
    spec = manifest["spec"]
    records = H.load(out / "00/records.json")
    rows = []
    for record, policy, obj, query, unit in itertools.product(
        records, range(5), range(5), range(6), range(4)
    ):
        address = (record["history"], policy, obj, query, unit)
        rows.append(
            {
                **record,
                "policy_name": spec["policies"][policy],
                "object": spec["objects"][obj],
                "query": spec["targets"][query],
                "unit": spec["units"][unit],
                **{
                    p: float(values[p][address])
                    for p in (
                        "observed",
                        "reference",
                        "policy",
                        "instrument",
                        "author_choice",
                    )
                },
            }
        )
    contrasts = []
    index = {(r["slot"], r["recipe"], r["seed"]): r["history"] for r in records}
    for slot, seed, pair, policy, obj, query, unit in itertools.product(
        ("A", "B"),
        spec["seeds"],
        ((1, 2), (3, 0), (5, 4)),
        range(5),
        range(5),
        range(6),
        range(4),
    ):
        left, right = index[(slot, pair[0], seed)], index[(slot, pair[1], seed)]
        address = (policy, obj, query, unit)
        differences = {
            p: float(values[p][left][address] - values[p][right][address])
            for p in ("observed", "reference", "policy")
        }
        contrasts.append(
            {
                "slot": slot,
                "seed": seed,
                "recipes": f"{pair[0]}-{pair[1]}",
                "policy_name": spec["policies"][policy],
                "object": spec["objects"][obj],
                "query": spec["targets"][query],
                "unit": spec["units"][unit],
                **differences,
            }
        )
    delta = values["observed"][:, 3] - values["observed"][:, 4]
    summary_rows = []
    for slot, obj, unit in itertools.product(("A", "B"), range(5), range(4)):
        selected = [r["history"] for r in records if r["slot"] == slot]
        x = delta[selected, obj, :, unit]
        summary_rows.append(
            {
                "slot": slot,
                "object": spec["objects"][obj],
                "unit": spec["units"][unit],
                "coupled_pair_RMS": float(np.sqrt(np.mean(x * x))),
                "coupled_pair_max_abs": float(np.max(abs(x))),
                "positive": int((x > spec["arithmetic_tolerance"]).sum()),
                "negative": int((x < -spec["arithmetic_tolerance"]).sum()),
                "zero_at_arithmetic_tolerance": int(
                    (abs(x) <= spec["arithmetic_tolerance"]).sum()
                ),
                "cells": int(x.size),
            }
        )
    dest = out / "02"
    dest.mkdir()
    H.table(dest / "policy_accounts.csv", rows)
    H.table(dest / "author_contrasts.csv", contrasts)
    H.table(dest / "coupling_summaries.csv", summary_rows)
    H.write(
        dest / "summary.json",
        {
            "state": "ANALYZED_PENDING_VERIFICATION",
            "rows": [len(rows), len(contrasts), len(summary_rows)],
            "histories": len(records),
            "events_per_history": spec["base_events"] * 2,
            "complete_native_and_four_actions": True,
            "new_forwards": 0,
            "fits": 0,
            "original_scope_complete": False,
            "analysis_seconds": time.monotonic() - start,
        },
    )
    H.write(dest / "output_seal.json", H.seals(out, dest))
    print("ANALYZED HM118", len(rows), len(contrasts), flush=True)


def verify(out):
    """Check direct event sums, equal margins and independent four-corner effects."""
    manifest = check(out)
    spec, values = manifest["spec"], read_scores(out)
    H.verify_hashes(out, H.load(out / "02/output_seal.json"))
    with np.load(out / "00/kernels.npz", allow_pickle=False) as stream:
        kernels = stream["kernels"]
    with np.load(out / "00/ledgers_instrument.npz", allow_pickle=False) as stream:
        counts, owners = stream["joint_counts"], stream["evidence_owner"]
    with np.load(out / "01/author_scores.npz", allow_pickle=False) as stream:
        author = {p: stream[p] for p in stream.files}
    errors = {
        p: 0.0
        for p in (
            "evidence_margins",
            "group_focus_margins",
            "direct_observed",
            "direct_reference",
            "closure",
            "four_corners",
            "provenance_score",
            "provenance_accounts",
            "source_export",
            "CSV",
        )
    }
    with np.load(
        ROOT / spec["source"] / "02/accounts.npz", allow_pickle=False
    ) as stream:
        raw = stream["raw"]
    for owner in range(2):
        actual = (
            raw[:, :, :, :, :, owner, owner]
            .transpose(0, 2, 3, 4, 1, 5, 6)
            .reshape(5, 24, 2, 6, 4)
        )
        errors["source_export"] = max(
            errors["source_export"],
            float(np.max(abs(kernels[owner, 0] - actual[0]))),
            float(np.max(abs(kernels[owner, 1:] - (actual[0] - actual[1:])))),
        )
    for h, policy in itertools.product(range(len(counts)), range(5)):
        joint, selected = counts[h, policy], kernels[owners[h]]
        n = joint.sum()
        direct, reference = np.zeros((5, 6, 4)), np.zeros((5, 6, 4))
        for event, focus in itertools.product(range(24), range(2)):
            direct += joint[event, focus] / n * selected[:, event, focus]
            reference += joint[event].sum() / (2 * n) * selected[:, event, focus]
        errors["direct_observed"] = max(
            errors["direct_observed"],
            float(np.max(abs(direct - values["observed"][h, policy]))),
        )
        errors["direct_reference"] = max(
            errors["direct_reference"],
            float(np.max(abs(reference - values["reference"][h, policy]))),
        )
        errors["evidence_margins"] = max(
            errors["evidence_margins"],
            float(np.max(abs(joint.sum(axis=1) - counts[h, 2].sum(axis=1)))),
        )
        if policy in (3, 4):
            grouped, balanced = (
                joint.reshape(12, 2, 2).sum(axis=1),
                counts[h, 2].reshape(12, 2, 2).sum(axis=1),
            )
            errors["group_focus_margins"] = max(
                errors["group_focus_margins"], float(np.max(abs(grouped - balanced)))
            )
    for h in range(len(counts)):
        direct = M.direct_pair(
            kernels[owners[h]], counts[h, 2].sum(axis=1), spec["coupling_divisor"]
        )
        errors["four_corners"] = max(
            errors["four_corners"],
            float(
                np.max(
                    abs(direct - (values["observed"][h, 3] - values["observed"][h, 4]))
                )
            ),
        )
    errors["closure"] = float(
        np.max(
            abs(
                values["observed"]
                - values["reference"]
                - values["instrument"]
                - values["author_choice"]
            )
        )
    )
    errors["provenance_score"] = float(
        np.max(abs(values["observed"] - author["observed"]))
    )
    errors["provenance_accounts"] = float(
        max(
            np.max(abs(values["instrument"] - author["author_choice"])),
            np.max(abs(author["instrument"])),
            np.max(abs(values["author_choice"])),
        )
    )
    if bool(values["neural_runtime_loaded"]) or bool(author["neural_runtime_loaded"]):
        raise ValueError("Cold scorer loaded a neural runtime")
    import csv

    with (out / "02/policy_accounts.csv").open(newline="") as stream:
        table = list(csv.DictReader(stream))
    expected = len(counts) * 5 * 5 * 6 * 4
    if len(table) != expected:
        raise ValueError("Full policy/target/unit support missing")
    records = H.load(out / "00/records.json")
    for row, (h, policy, obj, query, unit) in zip(
        table,
        itertools.product(range(len(counts)), range(5), range(5), range(6), range(4)),
        strict=True,
    ):
        metadata = {
            **records[h],
            "policy_name": spec["policies"][policy],
            "object": spec["objects"][obj],
            "query": spec["targets"][query],
            "unit": spec["units"][unit],
        }
        if any(row[key] != str(value) for key, value in metadata.items()):
            raise ValueError("Scientific address or policy name changed in table")
        for field in ("observed", "reference", "policy", "instrument", "author_choice"):
            errors["CSV"] = max(
                errors["CSV"],
                abs(
                    float(row[field])
                    - float(values[field][h, policy, obj, query, unit])
                ),
            )
    lookup = {(r["slot"], r["recipe"], r["seed"]): r["history"] for r in records}
    with (out / "02/author_contrasts.csv").open(newline="") as stream:
        contrasts = list(csv.DictReader(stream))
    cases = list(
        itertools.product(
            ("A", "B"),
            spec["seeds"],
            ((1, 2), (3, 0), (5, 4)),
            range(5),
            range(5),
            range(6),
            range(4),
        )
    )
    if len(contrasts) != len(cases):
        raise ValueError("Author comparison support missing")
    for row, (slot, seed, pair, policy, obj, query, unit) in zip(
        contrasts, cases, strict=True
    ):
        metadata = {
            "slot": slot,
            "seed": seed,
            "recipes": f"{pair[0]}-{pair[1]}",
            "policy_name": spec["policies"][policy],
            "object": spec["objects"][obj],
            "query": spec["targets"][query],
            "unit": spec["units"][unit],
        }
        if any(row[key] != str(value) for key, value in metadata.items()):
            raise ValueError("Author comparison scientific address changed")
        left, right = lookup[(slot, pair[0], seed)], lookup[(slot, pair[1], seed)]
        for field in ("observed", "reference", "policy"):
            value = (
                values[field][left, policy, obj, query, unit]
                - values[field][right, policy, obj, query, unit]
            )
            errors["CSV"] = max(errors["CSV"], abs(float(row[field]) - float(value)))
    if max(errors.values()) > spec["arithmetic_tolerance"]:
        raise ValueError(f"Finite policy replay failed:{errors}")
    H.write(
        out / "verification.json",
        {
            "status": "VERIFIED_FINITE_HM118_NOT_GLOBAL",
            "errors": errors,
            "new_forwards": 0,
            "new_fits": 0,
            "same_agent_not_third_party": True,
            "original_scope_complete": False,
        },
    )
    print("VERIFIED HM118", errors, flush=True)


def main():
    """Execute one finite phase; never automatically retry or renew model budgets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "cold", "analyze", "verify"))
    parser.add_argument("--output", type=Path, default=ROOT / H.load(SPEC)["output"])
    args = parser.parse_args()
    globals()[args.stage](args.output.resolve())


if __name__ == "__main__":
    main()
