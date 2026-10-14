#!/usr/bin/env python3
"""Run actual controlled-text inference for original T2/T3 with frozen sources."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from grain_research.gptpro_foundation_v3 import f16_dynamic_text as model
from grain_research.gptpro_foundation_v3.f14_metadata import sha256_file
from scripts.run_grain_gptpro_f16_t1_text import (
    evidence_cards,
    material_bank,
    metrics,
    numeric_difference,
    render,
    save,
    write_inputs,
)
from scripts.run_grain_gptpro_f16_t2 import group

ROOT = Path(__file__).resolve().parents[1]


def panel(inputs, bundle, arm="hybrid_joint", evaluator=None, scramble=0):
    """Bound working memory while keeping per-author calculation unchanged."""
    return [
        row
        for i in range(0, len(inputs), 48)
        for row in model.score_panel(
            inputs[i : i + 48], bundle, arm, evaluator, scramble
        )
    ]


def render_dynamic(groups, bank, role, seed):
    """Reuse the controlled-text renderer, preserving the F05 clock and conditions."""
    groups = {a: rows[:48] for a, rows in groups.items()}
    inputs, evaluator = render(groups, bank, role, seed)
    for raw in inputs:
        raw["target_time"] = 47
        raw["world_contract"] = model.CONTRACT
    return inputs, evaluator


def cold(directory, bundle, inputs, name):
    """Independent process has only the self-contained bundle and observed history."""
    source = directory / f"{name}_inputs.jsonl"
    dest = directory / f"{name}_cold_scores.jsonl"
    write_inputs(source, inputs)
    with (directory / f"{name}_cold.log").open("w") as stream:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "grain_research.gptpro_foundation_v3.f16_dynamic_text",
                "--bundle",
                str(bundle.resolve()),
                "--sha256",
                sha256_file(bundle),
                "--input",
                str(source.resolve()),
                "--output",
                str(dest.resolve()),
            ],
            cwd=directory,
            env={
                **os.environ,
                "PYTHONPATH": str(ROOT),
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return [json.loads(x) for x in dest.read_text().splitlines()]


def discrepancy(left, right):
    """Compare target means and SDs without any latent or future outcomes."""
    return {
        k: max(
            abs(
                (a["targets"][t]["score"] - b["targets"][t]["score"])
                if k == "score"
                else (
                    np.sqrt(a["targets"][t]["variance"])
                    - np.sqrt(b["targets"][t]["variance"])
                )
            )
            for a, b in zip(left, right, strict=True)
            for t in ("T2", "T3")
        )
        for k in ("score", "sd")
    }


def history_controls(inputs, bundle):
    """Execute revisions, aliases, withdrawals and clock-preserving missingness."""
    x = copy.deepcopy(inputs[0])
    base = model.cold_scores([x], bundle)[0]
    alias = copy.deepcopy(x)
    last = alias["history"][-1]["record_id"]
    alias["history"].append(
        {
            "op": "alias",
            "record_id": "extra_alias",
            "parent_id": last,
            "recorded_at": 47,
        }
    )
    alias["history"].append(
        {
            "op": "summary",
            "record_id": "extra_summary",
            "parents": [last],
            "recorded_at": 47,
        }
    )
    withdrawn = copy.deepcopy(alias)
    withdrawn["history"].append(
        {"op": "retract", "record_id": "withdraw", "parent_id": last, "recorded_at": 47}
    )
    revised = copy.deepcopy(alias)
    row = copy.deepcopy(x["history"][-1]["observation"])
    alternatives = sorted(bundle["materials"])
    row["material_id"] = next(k for k in alternatives if k != row["material_id"])
    revised["history"].append(
        {
            "op": "revise",
            "record_id": "revision",
            "parent_id": last,
            "recorded_at": 47,
            "observation": row,
        }
    )
    direct_revision = copy.deepcopy(x)
    direct_revision["history"][-1]["observation"] = row
    versions = {
        "base": x,
        "alias_summary": alias,
        "withdraw": withdrawn,
        "direct_delete": dict(x, history=x["history"][:-1]),
        "revise": revised,
        "direct_revision": direct_revision,
        "missing_tail4": dict(x, history=x["history"][:-4]),
        "empty": dict(x, history=[]),
    }
    out = model.cold_scores(list(versions.values()), bundle)
    results = dict(zip(versions, out, strict=True))
    assert (
        numeric_difference(base["targets"], results["alias_summary"]["targets"]) < 1e-10
    )
    for a, b in [("withdraw", "direct_delete"), ("revise", "direct_revision")]:
        assert numeric_difference(results[a]["targets"], results[b]["targets"]) < 1e-10
    assert all(r["target_time"] == 47 for r in results.values())
    return {"inputs": versions, "scores": results}


def main():
    """Freeze source references before opening application; save all six arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    base = ROOT / "results/grain_gptpro_foundation_v3"
    seeds = [20260915] if args.smoke else [20260915, 20260916, 20260917]
    files = [
        Path(__file__),
        ROOT / "grain_research/gptpro_foundation_v3/f16_dynamic_text.py",
        ROOT / "tests/test_grain_gptpro_f16_dynamic_text.py",
        ROOT / "docs/GRAIN_F16_DYNAMIC_TEXT_PROTOCOL_V1.md",
        ROOT / "grain_research/gptpro_foundation_v3/f16_t1_text.py",
        ROOT / "grain_research/gptpro_foundation_v3/f16_t1_source.py",
        ROOT / "scripts/run_grain_gptpro_f16_t1_text.py",
        ROOT / "scripts/run_grain_gptpro_f16_t2.py",
    ]
    source_files = [
        base / "F08/nli_reader_r1/reader_scores.csv",
        base / "F11/grain_comment_reader_r1/reader_scores.csv",
    ]
    for seed in seeds:
        source_files += [
            base / f"F16/t2_cold_full_r1/{seed}/bundle.json",
            base / f"F16/t1_text_full_r1/{seed}/bundle.json",
            base / f"F05/source_full_r2/core_{seed}/source.jsonl",
        ]
    save(
        args.output / "protocol_before_run.json",
        {
            "code_hashes": {str(p): sha256_file(p) for p in files},
            "source_hashes": {str(p): sha256_file(p) for p in source_files},
            "seeds": seeds,
            "smoke": args.smoke,
            "qmc_power": 4 if args.smoke else 8,
            "source_components": 3 if args.smoke else 200,
            "arms": model.ARMS,
            "target_time": 47,
            "future_conditions": model.PLAN,
            "new_neural_calls": 0,
            "source_refits": 0,
            "numerical_checks": "12fixedIDs/all6arms second scramble;>.02 saved1024diagnostic;ADF grid in separate audit",
        },
    )
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    for p in files:
        (snapshot / p.name).write_bytes(p.read_bytes())
    bank = material_bank(base)
    ready = []
    for seed in seeds:
        folder = args.output / str(seed)
        folder.mkdir()
        old = json.loads((base / f"F16/t2_cold_full_r1/{seed}/bundle.json").read_text())
        text = json.loads(
            (base / f"F16/t1_text_full_r1/{seed}/bundle.json").read_text()
        )
        archive = base / f"F05/source_full_r2/core_{seed}"
        if sha256_file(archive / "source.jsonl") != old["source_hash"]["source"]:
            raise ValueError("source archive changed")
        n = 3 if args.smoke else 200
        bundle = {
            "version": model.VERSION,
            "scorer_code_hash": model.code_hash(),
            "world_contract": model.CONTRACT,
            "future_conditions": model.PLAN,
            "target_ids": ["T2", "T3"],
            "object_classes": ["dynamic_state", "future_response"],
            "seed": seed,
            "qmc_power": 4 if args.smoke else 8,
            "source_components": n,
            "source_models": [x["model"] for x in old["source_models"][:n]],
            "channel_draws": {k: v[:n] for k, v in text["channel_draws"].items()},
            "materials": text["materials"],
            "source_hash": old["source_hash"],
            "source_bundle_sha256": sha256_file(
                base / f"F16/t2_cold_full_r1/{seed}/bundle.json"
            ),
            "reader_bundle_sha256": sha256_file(
                base / f"F16/t1_text_full_r1/{seed}/bundle.json"
            ),
            "reader_hash": hashlib.sha256(
                json.dumps(
                    {"materials": text["materials"], "channels": text["channel_draws"]},
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
            "unit": "numeric source-calibrated latent response units, not identified by polarity text alone",
            "reference": {t: {"mu": 0.0, "sigma": 1.0} for t in ("T2", "T3")},
            "qualification_rate": 0.5,
            "observed_role": "controlled synthetic text from archived F05 responses",
            "support_rule": "empty is prior-only; missing events retain requested clock",
            "error_model": "ADF state projection + prior-QMC parameter posterior + cut source/reader mixture",
            "invalidating_result": "reader/eligibility/transition mismatch or material approximation error",
            "explanation_rule": "text/history/clock/condition evidence actions, not psychological naming",
        }
        groups = group(archive / "source.jsonl", "source")
        if len(groups) != 300:
            raise ValueError("expected full source population")
        if args.smoke:
            groups = dict(sorted(groups.items())[:8])
        inputs, evaluator = render_dynamic(groups, bank, "source", seed)
        scores = panel(inputs, bundle)
        for target in ("T2", "T3"):
            v = np.array([x["targets"][target]["score"] for x in scores])
            bundle["reference"][target] = {
                "mu": float(v.mean()),
                "sigma": float(v.std(ddof=1)),
            }
            for x in scores:
                x["targets"][target]["reference_score"] = (
                    x["targets"][target]["score"] - v.mean()
                ) / v.std(ddof=1)
        bundle["source_reference_authors"] = len(groups)
        save(folder / "bundle.json", bundle)
        write_inputs(folder / "source_inputs.jsonl", inputs)
        save(folder / "source_scores.json", scores)
        save(folder / "source_rendering_evaluator.json", evaluator)
        ready.append((seed, folder, bundle, archive, set(groups)))
        print(
            json.dumps(
                {
                    "stage": "source_frozen",
                    "seed": seed,
                    "authors": len(groups),
                    "seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
    flat, cards, numerical = [], [], []
    for seed, folder, bundle, archive, source_ids in ready:
        groups = group(archive / "application.jsonl", "application")
        if len(groups) != 200 or source_ids.intersection(groups):
            raise ValueError("application population/partition mismatch")
        if args.smoke:
            groups = dict(sorted(groups.items())[:4])
        inputs, evaluator = render_dynamic(groups, bank, "application", seed)
        save(folder / "application_rendering_evaluator.json", evaluator)
        save(
            folder / "application_hashes.json",
            {
                "observations": sha256_file(archive / "application.jsonl"),
                "truth": sha256_file(archive / "truth.json"),
            },
        )
        ev = {(r["author_id"], r["event_id"]): r for r in evaluator}
        outputs = {}
        for arm in model.ARMS:
            outputs[arm] = panel(inputs, bundle, arm, ev)
            save(folder / f"{arm}_scores.json", outputs[arm])
            print(
                json.dumps(
                    {
                        "stage": "arm_done",
                        "seed": seed,
                        "arm": arm,
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
        primary = outputs["hybrid_joint"]
        c = cold(folder, folder / "bundle.json", inputs, "application")
        error = numeric_difference(primary, c)
        if error > 1e-10:
            raise ValueError("cold scoring mismatch")
        checks = []
        for arm in model.ARMS:
            sub = inputs[:12]
            second = panel(sub, bundle, arm, ev, scramble=1)
            delta = discrepancy(outputs[arm][: len(sub)], second)
            check = {
                "seed": seed,
                "arm": arm,
                "authors": len(sub),
                "scramble256": delta,
            }
            if max(delta.values()) > 0.02 and not args.smoke:
                fine = dict(bundle, qmc_power=10)
                f0, f1 = (
                    panel(sub, fine, arm, ev),
                    panel(sub, fine, arm, ev, scramble=1),
                )
                check["scramble1024"] = discrepancy(f0, f1)
                check["primary_vs1024"] = discrepancy(outputs[arm][: len(sub)], f0)
                save(folder / f"qmc1024_{arm}.json", {"scramble0": f0, "scramble1": f1})
            checks.append(check)
        numerical.extend(checks)
        save(folder / "qmc_checks.json", checks)
        prepared = evidence_cards(inputs, bank, bundle)
        scored = model.cold_scores([x for _, x in prepared], bundle)
        card_cold = cold(
            folder, folder / "bundle.json", [x for _, x in prepared], "cards"
        )
        if numeric_difference(scored, card_cold) > 1e-10:
            raise ValueError("card cold mismatch")
        for (name, x), result in zip(prepared, scored, strict=True):
            cards.append(
                {
                    "seed": seed,
                    "name": name,
                    "author_id": x["author_id"],
                    "result": result,
                }
            )
        save(folder / "history_controls.json", history_controls(inputs, bundle))
        # All six arms, numerical settings and cold outputs precede evaluation truth.
        truth = {
            r["author_id"]: r
            for r in json.loads((archive / "truth.json").read_text())
            if r["role"] == "application"
        }
        for i, x in enumerate(inputs):
            actual = truth[x["author_id"]]
            future = groups[x["author_id"]][48:52]
            if [r["t"] for r in future] != list(range(48, 52)):
                raise ValueError("wrong future target clock")
            outcomes = {
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
                            "truth": outcomes[target],
                            **{
                                k: s[k] for k in ("score", "variance", "lower", "upper")
                            },
                        }
                    )
        save(
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
    frame.to_csv(args.output / "scores.csv", index=False)
    tables, contrasts = [], []
    for target, subset in frame.groupby("target"):
        table = metrics(subset)
        table["target"] = target
        tables.append(table)
        for arm in model.ARMS[1:]:
            pair = subset[subset.arm.eq("hybrid_joint")].merge(
                subset[subset.arm.eq(arm)],
                on=["seed", "author_id"],
                suffixes=("_p", "_r"),
                validate="one_to_one",
            )
            delta = (pair.score_p - pair.truth_p) ** 2 - (
                pair.score_r - pair.truth_r
            ) ** 2
            rng, boot = np.random.default_rng(165), np.zeros(2000)
            for seed in seeds:
                x = delta[pair.seed.eq(seed)].to_numpy()
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
    table.to_csv(args.output / "metrics_by_seed.csv", index=False)
    table.groupby(["target", "arm"])[
        ["mse", "bias", "r", "coverage", "width"]
    ].mean().to_csv(args.output / "metrics.csv")
    pd.DataFrame(contrasts).to_csv(args.output / "paired_contrasts.csv", index=False)
    save(args.output / "evidence_cards.json", cards)
    save(
        args.output / "summary.json",
        {
            "status": "DYNAMIC_TEXT_PRIMARY_EXECUTED_APPROXIMATION_AUDIT_REQUIRED",
            "smoke": args.smoke,
            "application_authors": len(frame) // 12,
            "targets": 2,
            "arms": 6,
            "cards_per_target": len(cards),
            "numerical_checks": numerical,
            "seconds": time.monotonic() - started,
            "scope": "controlled texts on SIM dynamic targets, not new human validity",
            "new_neural_calls": 0,
        },
    )
    artifacts = [
        p
        for p in args.output.rglob("*")
        if p.is_file()
        and not p.name.startswith("._")
        and "source_snapshot" not in p.parts
    ]
    save(
        args.output / "output_manifest.json",
        {str(p.relative_to(args.output)): sha256_file(p) for p in artifacts},
    )
    print(
        json.dumps({"stage": "complete", "seconds": time.monotonic() - started}),
        flush=True,
    )


if __name__ == "__main__":
    main()
