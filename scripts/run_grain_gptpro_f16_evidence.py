#!/usr/bin/env python3
"""Run24 numeric evidence cards against existing, unchanged T1/T2/T3 bundles."""

import argparse
import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from grain_research.gptpro_foundation_v3 import f16_evidence_ledger as ledger
from grain_research.gptpro_foundation_v3.f16_cutoff_audit import dense_moments

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/grain_gptpro_foundation_v3"
CONFIG = ROOT / "configs/grain_f16_evidence_v1.json"


def digest(path):
    """Bind an actual file rather than trusting a legacy completion flag."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    """Atomically record a finite checkpoint without replacing an older run."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def load_materials(target, config):
    """Read the authorized synthetic application prefix, never target truth."""
    seed = config["seed"]
    roots = {
        "T1": "t1_source_full_r1",
        "T2": "t2_cold_full_r1",
        "T3": "t3_application_full_r1",
    }
    old = RESULTS / "F16" / roots[target] / str(seed)
    basefile = old / "bundle.json"
    base = json.loads(basefile.read_text())
    manifestfile = old / "application_manifest.json"
    manifest = json.loads(manifestfile.read_text())
    bundle_manifest = (
        old / "protocol_before_application.json" if target == "T3" else manifestfile
    )
    expected_bundle = json.loads(bundle_manifest.read_text())["bundle_sha256"]
    if digest(basefile) != expected_bundle:
        raise ValueError("source bundle is not the archived application bundle")
    if base["personal_draws"] != config["personal_draws"]:
        raise ValueError("personal integration budget changed")
    count = len(base["source_models"]) if target != "T3" else len(base["ensemble"])
    if count != config["source_bootstraps"]:
        raise ValueError("source ensemble incomplete")
    if target == "T1":
        appfile = RESULTS / "F04/source_full_r1" / f"L_{seed}" / "application.jsonl"
        expected = manifest["application_sha256"]
    else:
        path_manifest = json.loads(
            (
                RESULTS
                / "F16/t3_application_full_r1"
                / str(seed)
                / "application_manifest.json"
            ).read_text()
        )
        appfile = Path(path_manifest["file"])
        expected = path_manifest["sha256"]
        if target == "T2" and manifest["input_hash"] != expected:
            raise ValueError("dynamic material identity mismatch")
    if digest(appfile) != expected:
        raise ValueError("application material changed")
    requested = sorted({c["author_index"] for c in config["cases"]})
    groups = {manifest["author_order"][i]: [] for i in requested}
    prefix = config["prefix_by_target"][target]
    for number, line in enumerate(appfile.open(), start=1):
        row = json.loads(line)
        if row["author_id"] in groups and row["t"] < prefix:
            groups[row["author_id"]].append((row, f"{appfile}#L{number}"))
    for rows in groups.values():
        if [row["t"] for row, _ in rows] != list(range(prefix)):
            raise ValueError("prefix has wrong order or missing original observations")
    return (
        base,
        groups,
        manifest["author_order"],
        {
            "base_bundle": str(basefile),
            "base_bundle_sha256": digest(basefile),
            "application": str(appfile),
            "application_sha256": expected,
            "application_manifest": str(manifestfile),
            "manifest_sha256": digest(manifestfile),
            "bundle_manifest": str(bundle_manifest),
            "bundle_manifest_sha256": digest(bundle_manifest),
        },
    )


def original_input(rows, bundle):
    """Wrap actual numeric rows with explicit assumed-actual report metadata."""
    return {
        "author_id": rows[0][0]["author_id"],
        "world": bundle["world"],
        "reader_hash": bundle["reader_hash"],
        "target_time": rows[-1][0]["t"],
        "history": [
            {
                "record_id": f"r{i:03d}",
                "op": "new",
                "recorded_at": row["t"],
                "material_ref": ref,
                "qualification": "actual",
                "observation": copy.deepcopy(row),
            }
            for i, (row, ref) in enumerate(rows)
        ],
    }


def case_input(original, case, config, case_ref):
    """Apply declared edits with explicit ancestry and a fixed scoring cutoff."""
    raw = copy.deepcopy(original)
    originals = copy.deepcopy(raw["history"])
    name = case["name"]
    edits = []

    def add(kind, **fields):
        index = len(edits)
        identity = f"{name}:edit:{index}"
        record = {
            "record_id": identity,
            "op": kind,
            "recorded_at": raw["target_time"],
            "material_ref": f"{case_ref}#/history/{len(raw['history'])}",
            **fields,
        }
        edits.append(identity)
        raw["history"].append(record)

    if name == "full_history":
        add("alias", parent_id=originals[0]["record_id"])
        add("alias", parent_id=originals[-1]["record_id"])
        add("summary", parents=[originals[0]["record_id"], originals[-1]["record_id"]])
        row = {
            **originals[-1]["observation"],
            "y": originals[-1]["observation"]["y"] + config["correction_delta"],
        }
        add(
            "revise",
            parent_id=originals[-1]["record_id"],
            observation=row,
            qualification="negated",
        )
    elif name in ("missing_plus", "missing_minus", "selection_high"):
        for record in originals:
            row = record["observation"]
            remove = (
                row["z"] == 1
                if name == "missing_plus"
                else row["z"] == -1
                if name == "missing_minus"
                else row["y"] < config["selection_threshold"]
            )
            if remove:
                add("retract", parent_id=record["record_id"])
    elif name in ("late_offset", "quoted_tail", "hypothetical_all"):
        selected = (
            originals
            if name == "hypothetical_all"
            else originals[-config["tail_count"] :]
        )
        for record in selected:
            row = copy.deepcopy(record["observation"])
            if name == "late_offset":
                row["y"] += config["late_offset_delta"]
            qualification = {
                "late_offset": "actual",
                "quoted_tail": "quoted",
                "hypothetical_all": "hypothetical",
            }[name]
            add(
                "revise",
                parent_id=record["record_id"],
                observation=row,
                qualification=qualification,
            )
    elif name != "full_plain":
        raise ValueError("unknown frozen case")
    return raw


def explanation(name, target):
    """Name the operation and next measurement action, not a psychological trait."""
    descriptions = {
        "full_plain": (
            "Unedited numeric prefix.",
            "Use the frozen target and reference; do not infer new calibration.",
        ),
        "full_history": (
            "Aliases/summary do not add information; final report corrected by +0.25.",
            "Replace the old report and flag stale derived summaries; correction is not a second observation.",
        ),
        "missing_plus": (
            "All +1 condition reports are withdrawn.",
            "Report missing condition support and source borrowing; obtain matching condition evidence if needed.",
        ),
        "missing_minus": (
            "All -1 condition reports are withdrawn.",
            "Report missing condition support and source borrowing; do not turn missing into zero.",
        ),
        "selection_high": (
            "Only y>=0 remains. This is outcome-dependent availability.",
            "Keep opportunity denominator/selection mechanism visible; working ignorable score is not MNAR repair.",
        ),
        "late_offset": (
            "Last4 report values receive a known +1 observation-channel offset.",
            "Flag channel shift versus genuine state as alternative explanations; changed score alone cannot separate them.",
        ),
        "quoted_tail": (
            "Last4 are explicitly reclassified as quoted, not self observations.",
            "Exclude from self likelihood, keep provenance, propagate state to the unchanged cutoff.",
        ),
        "hypothetical_all": (
            "All reports are explicitly reclassified as hypothetical.",
            "Return source-prior output marked no measured self evidence, not an observed neutral score.",
        ),
    }
    operation, action = descriptions[name]
    return {
        "target": target,
        "operation": operation,
        "action": action,
        "qualification_source": "supplied_simulation_metadata_not_language_reader",
        "effect_kind": "finite_same_target_edit_difference_not_additive_factor_contribution",
        "uncertainty": "Separate target intervals;no interval for paired edit effect and no new calibration claim.",
    }


def run_target(task):
    """Score fixed cases with cached sources and replay them in a cold process."""
    target, root, config, resume = task
    started = time.monotonic()
    out = root / target
    out.mkdir(exist_ok=resume)
    base, groups, author_order, provenance = load_materials(target, config)
    bundle = ledger.build_bundle(base)
    protocol = {
        "config": config,
        "target": target,
        "inputs": provenance,
        "ledger_hash": ledger.code_hash(),
        "runner_hash": digest(__file__),
    }
    if (out / "protocol.json").exists() and json.loads(
        (out / "protocol.json").read_text()
    ) != protocol:
        raise ValueError("resume code/config/input mismatch")
    save(out / "protocol.json", protocol)
    save(out / "bundle.json", bundle)
    (out / "cards").mkdir(exist_ok=True)
    (out / "inputs").mkdir(exist_ok=True)
    (out / "scores").mkdir(exist_ok=True)
    originals, scored = {}, {}
    for index in sorted({c["author_index"] for c in config["cases"]}):
        originals[index] = original_input(groups[author_order[index]], bundle)
    inputs = [(f"before_{i:03d}", v) for i, v in originals.items()]
    inputs += [
        (
            c["name"],
            case_input(
                originals[c["author_index"]],
                c,
                config,
                str(out / "inputs" / f"{c['name']}.json"),
            ),
        )
        for c in config["cases"]
    ]
    for key, raw in inputs:
        save(out / "inputs" / f"{key}.json", raw)
        file = out / "scores" / f"{key}.json"
        if file.exists():
            entry = json.loads(file.read_text())
            if entry["input_hash"] != ledger.clock.object_hash(raw):
                raise ValueError("resume evidence changed")
        else:
            checks = []

            # This callback checks all personal draws in predetermined source fits.
            def audit(
                bootstrap, parameters, noise, mean, variance, raw=raw, checks=checks
            ):
                if bootstrap in config["dense_audit_source_indices"]:
                    selected = ledger.replay(raw, bundle)["records"]
                    dm, dv = dense_moments(
                        parameters, selected, noise, raw["target_time"], target
                    )
                    np.testing.assert_allclose(
                        mean, dm, atol=config["atol"], rtol=config["rtol"]
                    )
                    np.testing.assert_allclose(
                        variance, dv, atol=config["atol"], rtol=config["rtol"]
                    )
                    checks.append(
                        {
                            "bootstrap": bootstrap,
                            "draws": len(parameters),
                            "max_mean_error": float(np.max(np.abs(mean - dm))),
                            "max_variance_error": float(np.max(np.abs(variance - dv))),
                        }
                    )

            result = ledger.score_evidence(raw, bundle, audit_callback=audit)
            entry = {
                "input_hash": ledger.clock.object_hash(raw),
                "output": result,
                "dense_checks": checks,
            }
            save(file, entry)
        scored[key] = entry["output"]
        print(
            f"evidence {target} {key} n={entry['output']['ledger']['eligible_events']}",
            flush=True,
        )
    coldpath = out / "cold_scores.jsonl"
    if not coldpath.exists():
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            (tmp / "bundle.json").write_bytes((out / "bundle.json").read_bytes())
            (tmp / "inputs.jsonl").write_text(
                "".join(json.dumps(raw) + "\n" for _, raw in inputs)
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    ledger.__name__,
                    "--bundle",
                    "bundle.json",
                    "--sha256",
                    digest(tmp / "bundle.json"),
                    "--input",
                    "inputs.jsonl",
                    "--output",
                    "scores.jsonl",
                ],
                cwd=tmp,
                check=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(ROOT),
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                },
            )
            coldpath.write_bytes((tmp / "scores.jsonl").read_bytes())
    cold = [json.loads(s) for s in coldpath.read_text().splitlines()]
    if cold != [scored[key] for key, _ in inputs]:
        raise ValueError("cold scores differ from actual warm outputs")
    effects = []
    for case in config["cases"]:
        name, index = case["name"], case["author_index"]
        before, after = scored[f"before_{index:03d}"], scored[name]
        b, a = before["result"], after["result"]
        delta = (
            None
            if a["score"] is None or b["score"] is None
            else a["score"] - b["score"]
        )
        effect = {
            "target": target,
            "case": name,
            "category": case["category"],
            "author_index": index,
            "before_count": before["ledger"]["eligible_events"],
            "after_count": after["ledger"]["eligible_events"],
            "before_score": b["score"],
            "after_score": a["score"],
            "score_delta": delta,
            "before_variance": b["variance"],
            "after_variance": a["variance"],
            "prior_only": a["prior_only"],
            "stale_derived": len(after["ledger"]["stale_derived"]),
        }
        effects.append(effect)
        save(
            out / "cards" / f"{name}.json",
            {
                "card_id": f"{target}:{name}",
                "evidence_type": "SIM",
                "case": case,
                "author_id": originals[index]["author_id"],
                "target_time": originals[index]["target_time"],
                "original_materials": [
                    r["material_ref"] for r in originals[index]["history"]
                ],
                "before_input": str(out / "inputs" / f"before_{index:03d}.json"),
                "after_input": str(out / "inputs" / f"{name}.json"),
                "before": before,
                "after": after,
                "effect": effect,
                "interpretation": explanation(name, target),
                "selection_flag": "outcome_dependent_not_corrected"
                if name == "selection_high"
                else "fixed_designed_edit",
                "observation_unit": base.get("unit", "F04 response units"),
                "source_bundle_sha256": digest(out / "bundle.json"),
            },
        )
    save(
        out / "summary.json",
        {
            "target": target,
            "cards": len(effects),
            "cold_objects": len(cold),
            "elapsed_seconds": time.monotonic() - started,
            "effects": effects,
        },
    )
    return effects


def main():
    """Execute the frozen full case budget or an explicitly smaller smoke run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    config = json.loads(CONFIG.read_text())
    if args.smoke:
        config["cases"] = [
            c for c in config["cases"] if c["name"] in ("full_history", "quoted_tail")
        ]
        config["expected_cards"] = 6
    root.mkdir(parents=True, exist_ok=args.resume)
    protocol = {
        "config": config,
        "smoke": args.smoke,
        "config_sha256": digest(CONFIG),
        "runner_sha256": digest(__file__),
        "ledger_hash": ledger.code_hash(),
        "created_at_unix": None,
    }
    if (root / "protocol.json").exists() and json.loads(
        (root / "protocol.json").read_text()
    ) != protocol:
        raise ValueError("changed run protocol")
    save(root / "protocol.json", protocol)
    with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
        groups = list(
            pool.map(
                run_target,
                [(target, root, config, args.resume) for target in config["targets"]],
            )
        )
    rows = [row for group in groups for row in group]
    if len(rows) != config["expected_cards"]:
        raise ValueError("incomplete actual card set")
    with (root / "effects.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# F16 numeric evidence cards",
        "",
        "SIM only; metadata qualification, not natural-text inference.",
        "No new calibration or accuracy estimate. Source bundles unchanged.",
        "",
        "|Target|Case|Before n|After n|Score change|Prior only|",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in rows:
        delta = "null" if r["score_delta"] is None else f"{r['score_delta']:+.6f}"
        lines.append(
            f"|{r['target']}|{r['case']}|{r['before_count']}|{r['after_count']}|{delta}|{r['prior_only']}|"
        )
    (root / "EVIDENCE_CARDS.md").write_text("\n".join(lines) + "\n")
    save(
        root / "summary.json",
        {
            "cards": len(rows),
            "targets": config["targets"],
            "scope": "SIM_ONLY",
            "smoke": args.smoke,
            "F16_parent_complete": False,
            "source_refits": 0,
        },
    )


if __name__ == "__main__":
    main()
