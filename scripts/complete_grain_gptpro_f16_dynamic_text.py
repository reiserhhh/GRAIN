#!/usr/bin/env python3
"""Resume the frozen dynamic-text run after its numpy-scalar comparator failure."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from grain_research.gptpro_foundation_v3 import f16_dynamic_text as model
from grain_research.gptpro_foundation_v3.f14_metadata import sha256_file
from scripts.run_grain_gptpro_f16_dynamic_text import (
    ROOT,
    cold,
    discrepancy,
    panel,
    render_dynamic,
)
from scripts.run_grain_gptpro_f16_t1_text import (
    evidence_cards,
    material_bank,
    metrics,
    numeric_difference,
    save,
)
from scripts.run_grain_gptpro_f16_t2 import group


def json_difference(left, right):
    """Normalize numpy JSON scalars before the frozen recursive numeric comparison."""
    return numeric_difference(
        json.loads(json.dumps(left, allow_nan=False)),
        json.loads(json.dumps(right, allow_nan=False)),
    )


def retain(path, value):
    """Never overwrite a computed artifact; require exact semantic identity."""
    normalized = json.loads(json.dumps(value, allow_nan=False))
    if path.exists():
        if json.loads(path.read_text()) != normalized:
            raise ValueError(f"existing result differs: {path}")
    else:
        save(path, normalized)


def resume_cold(folder, bundle_path, inputs, name):
    """Read a completed cold process only when its exact inputs still match."""
    path = folder / f"{name}_cold_scores.jsonl"
    if path.exists():
        old = [
            json.loads(x)
            for x in (folder / f"{name}_inputs.jsonl").read_text().splitlines()
        ]
        if old != inputs:
            raise ValueError("cold input mismatch on resume")
        out = [json.loads(x) for x in path.read_text().splitlines()]
        if len(out) != len(inputs):
            raise ValueError("partial cold output")
        return out
    return cold(folder, bundle_path, inputs, name)


def controls(inputs, bundle):
    """Same fixed history operations, with a numeric rather than dtype comparison."""
    x = copy.deepcopy(inputs[0])
    alias = copy.deepcopy(x)
    parent = alias["history"][-1]["record_id"]
    alias["history"] += [
        {
            "op": "alias",
            "record_id": "extra_alias",
            "parent_id": parent,
            "recorded_at": 47,
        },
        {
            "op": "summary",
            "record_id": "extra_summary",
            "parents": [parent],
            "recorded_at": 47,
        },
    ]
    withdrawn = copy.deepcopy(alias)
    withdrawn["history"].append(
        {
            "op": "retract",
            "record_id": "withdraw",
            "parent_id": parent,
            "recorded_at": 47,
        }
    )
    revised = copy.deepcopy(alias)
    row = copy.deepcopy(x["history"][-1]["observation"])
    row["material_id"] = next(
        k for k in sorted(bundle["materials"]) if k != row["material_id"]
    )
    revised["history"].append(
        {
            "op": "revise",
            "record_id": "revision",
            "parent_id": parent,
            "recorded_at": 47,
            "observation": row,
        }
    )
    direct = copy.deepcopy(x)
    direct["history"][-1]["observation"] = row
    variants = {
        "base": x,
        "alias_summary": alias,
        "withdraw": withdrawn,
        "direct_delete": dict(x, history=x["history"][:-1]),
        "revise": revised,
        "direct_revision": direct,
        "missing_tail4": dict(x, history=x["history"][:-4]),
        "empty": dict(x, history=[]),
    }
    results = dict(
        zip(variants, model.cold_scores(list(variants.values()), bundle), strict=True)
    )
    singleton = model.cold_scores([x], bundle)[0]
    error = json_difference(singleton["targets"], results["base"]["targets"])
    for a, b in [
        ("base", "alias_summary"),
        ("withdraw", "direct_delete"),
        ("revise", "direct_revision"),
    ]:
        error = max(
            error, json_difference(results[a]["targets"], results[b]["targets"])
        )
    if error > 1e-10 or not all(r["target_time"] == 47 for r in results.values()):
        raise ValueError("actual numeric history invariance failed")
    return {"inputs": variants, "scores": results, "max_invariance_error": error}


def main():
    """Continue missing cells under unchanged source/code hashes and report scope."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    if (args.run / "summary.json").exists():
        raise ValueError("run already terminal")
    protocol = json.loads((args.run / "protocol_before_run.json").read_text())
    for key in ("code_hashes", "source_hashes"):
        for p, h in protocol[key].items():
            if sha256_file(p) != h:
                raise ValueError(f"frozen dependency changed: {p}")
    retain(
        args.run / "completion_protocol.json",
        {
            "code_hash": sha256_file(__file__),
            "original_protocol_hash": sha256_file(
                args.run / "protocol_before_run.json"
            ),
            "failure": "numeric_difference rejects numpy.float64 and compares exact equality; observed singleton/batch drift2.22e-16",
            "remedy": "JSON-normalize scalar containers; original1e-10 threshold/formula/bundle/saved scores unchanged",
            "resume_rule": "reuse verified norms, arms, inputs, cold results and numerical checks; compute only missing results",
        },
    )
    seeds = protocol["seeds"]
    base = ROOT / "results/grain_gptpro_foundation_v3"
    bank = material_bank(base)
    flat, cards, checks_all = [], [], []
    for seed in seeds:
        folder = args.run / str(seed)
        bundle = json.loads((folder / "bundle.json").read_text())
        model.validate_bundle(bundle)
        archive = base / f"F05/source_full_r2/core_{seed}"
        groups = group(archive / "application.jsonl", "application")
        if len(groups) != 200:
            raise ValueError("full application population required")
        inputs, evaluator = render_dynamic(groups, bank, "application", seed)
        retain(folder / "application_rendering_evaluator.json", evaluator)
        retain(
            folder / "application_hashes.json",
            {
                "observations": sha256_file(archive / "application.jsonl"),
                "truth": sha256_file(archive / "truth.json"),
            },
        )
        ev = {(r["author_id"], r["event_id"]): r for r in evaluator}
        outputs = {}
        for arm in model.ARMS:
            path = folder / f"{arm}_scores.json"
            reused = path.exists()
            outputs[arm] = (
                json.loads(path.read_text())
                if reused
                else panel(inputs, bundle, arm, ev)
            )
            retain(path, outputs[arm])
            if len(outputs[arm]) != len(inputs):
                raise ValueError("partial arm")
            print(
                json.dumps(
                    {
                        "stage": "arm",
                        "seed": seed,
                        "arm": arm,
                        "reused": reused,
                        "seconds": time.monotonic() - start,
                    }
                ),
                flush=True,
            )
        cold_scores = resume_cold(folder, folder / "bundle.json", inputs, "application")
        error = json_difference(outputs["hybrid_joint"], cold_scores)
        if error > 1e-10:
            raise ValueError("cold mismatch")
        if (folder / "qmc_checks.json").exists():
            checks = json.loads((folder / "qmc_checks.json").read_text())
        else:
            checks = []
            for arm in model.ARMS:
                sub = inputs[:12]
                second = panel(sub, bundle, arm, ev, scramble=1)
                delta = discrepancy(outputs[arm][:12], second)
                item = {"seed": seed, "arm": arm, "authors": 12, "scramble256": delta}
                if max(delta.values()) > 0.02:
                    fine = dict(bundle, qmc_power=10)
                    f0, f1 = (
                        panel(sub, fine, arm, ev),
                        panel(sub, fine, arm, ev, scramble=1),
                    )
                    item["scramble1024"] = discrepancy(f0, f1)
                    item["primary_vs1024"] = discrepancy(outputs[arm][:12], f0)
                    retain(
                        folder / f"qmc1024_{arm}.json",
                        {"scramble0": f0, "scramble1": f1},
                    )
                checks.append(item)
            retain(folder / "qmc_checks.json", checks)
        checks_all.extend(checks)
        prepared = evidence_cards(inputs, bank, bundle)
        scores = model.cold_scores([x for _, x in prepared], bundle)
        cold_cards = resume_cold(
            folder, folder / "bundle.json", [x for _, x in prepared], "cards"
        )
        if json_difference(scores, cold_cards) > 1e-10:
            raise ValueError("cold cards mismatch")
        for (name, x), result in zip(prepared, scores, strict=True):
            cards.append(
                {
                    "seed": seed,
                    "name": name,
                    "author_id": x["author_id"],
                    "result": result,
                }
            )
        retain(folder / "history_controls.json", controls(inputs, bundle))
        truth = {
            r["author_id"]: r
            for r in json.loads((archive / "truth.json").read_text())
            if r["role"] == "application"
        }
        for i, x in enumerate(inputs):
            actual = truth[x["author_id"]]
            future = groups[x["author_id"]][48:52]
            if [r["t"] for r in future] != list(range(48, 52)):
                raise ValueError("future clock mismatch")
            values = {
                "T2": actual["states"][47],
                "T3": float(
                    np.mean(
                        [
                            r["y"] + actual["b"] * (z - r["z"])
                            for z, r in zip(model.PLAN, future, strict=True)
                        ]
                    )
                ),
            }
            for arm, records in outputs.items():
                for target in ("T2", "T3"):
                    s = records[i]["targets"][target]
                    flat.append(
                        {
                            "seed": seed,
                            "author_id": x["author_id"],
                            "arm": arm,
                            "target": target,
                            "truth": values[target],
                            **{
                                k: s[k] for k in ("score", "variance", "lower", "upper")
                            },
                        }
                    )
        retain(
            folder / "cell_summary.json",
            {
                "seed": seed,
                "application_authors": len(inputs),
                "cold_warm_error": error,
                "minimum_component_ess": {
                    a: min(x["qmc_ess_min"] for x in rows)
                    for a, rows in outputs.items()
                },
            },
        )
    frame = pd.DataFrame(flat)
    frame.to_csv(args.run / "scores.csv", index=False)
    tables, contrasts = [], []
    for target, subset in frame.groupby("target"):
        table = metrics(subset)
        table["target"] = target
        tables.append(table)
        for arm in model.ARMS[1:]:
            pair = subset[subset.arm == "hybrid_joint"].merge(
                subset[subset.arm == arm],
                on=["seed", "author_id"],
                suffixes=("_p", "_r"),
                validate="one_to_one",
            )
            delta = (pair.score_p - pair.truth_p) ** 2 - (
                pair.score_r - pair.truth_r
            ) ** 2
            rng, boot = np.random.default_rng(165), np.zeros(2000)
            for seed in seeds:
                x = delta[pair.seed == seed].to_numpy()
                boot += x[rng.integers(0, len(x), (2000, len(x)))].mean(1) / len(seeds)
            lo, hi = np.quantile(boot, [0.025, 0.975])
            contrasts.append(
                {
                    "target": target,
                    "reference": arm,
                    "delta_mse": float(delta.mean()),
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                }
            )
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(args.run / "metrics_by_seed.csv", index=False)
    table.groupby(["target", "arm"])[
        ["mse", "bias", "r", "coverage", "width"]
    ].mean().to_csv(args.run / "metrics.csv")
    pd.DataFrame(contrasts).to_csv(args.run / "paired_contrasts.csv", index=False)
    retain(args.run / "evidence_cards.json", cards)
    retain(
        args.run / "summary.json",
        {
            "status": "DYNAMIC_TEXT_PRIMARY_EXECUTED_APPROXIMATION_AUDIT_REQUIRED",
            "smoke": False,
            "application_authors": len(frame) // 12,
            "targets": 2,
            "arms": 6,
            "cards_per_target": len(cards),
            "numerical_checks": checks_all,
            "completion_seconds": time.monotonic() - start,
            "scope": "controlled texts on SIM dynamic targets, not human validity",
            "new_neural_calls": 0,
            "recovery": "frozen source/scorer/formulas unchanged; numpy scalar comparison normalized; original stop retained",
        },
    )
    artifacts = [
        p
        for p in args.run.rglob("*")
        if p.is_file()
        and not p.name.startswith("._")
        and "source_snapshot" not in p.parts
    ]
    retain(
        args.run / "output_manifest.json",
        {str(p.relative_to(args.run)): sha256_file(p) for p in artifacts},
    )
    print(
        json.dumps({"stage": "complete", "seconds": time.monotonic() - start}),
        flush=True,
    )


if __name__ == "__main__":
    main()
