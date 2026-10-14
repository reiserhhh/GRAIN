#!/usr/bin/env python3
"""Post-result numerical tail check selected by ESS alone, without opening truth."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from grain_research.gptpro_foundation_v3 import f16_dynamic_text as model
from grain_research.gptpro_foundation_v3.f14_metadata import sha256_file
from scripts.run_grain_gptpro_f16_dynamic_text import discrepancy, panel
from scripts.run_grain_gptpro_f16_t1_text import save


def main():
    """Compare 256/1024/4096 parameter nodes for each arm's lowest-ESS author."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    original = json.loads((args.run / "protocol_before_run.json").read_text())
    save(
        args.output / "protocol.json",
        {
            "selection": "lowest saved component ESS per seed/arm;never truth/loss",
            "powers": [10, 12],
            "scrambles": [0, 1],
            "script_hash": sha256_file(__file__),
            "run_summary_hash": sha256_file(args.run / "summary.json"),
            "role": "post-result numerical diagnostic,not replacement of primary results",
        },
    )
    records = []
    for seed in original["seeds"]:
        root = args.run / str(seed)
        bundle = json.loads((root / "bundle.json").read_text())
        inputs = [
            json.loads(x)
            for x in (root / "application_inputs.jsonl").read_text().splitlines()
        ]
        evaluator = json.loads(
            (root / "application_rendering_evaluator.json").read_text()
        )
        ev = {(x["author_id"], x["event_id"]): x for x in evaluator}
        for arm in model.ARMS:
            old = json.loads((root / f"{arm}_scores.json").read_text())
            i = int(np.argmin([x["qmc_ess_min"] for x in old]))
            row = {
                "seed": seed,
                "arm": arm,
                "author_index": i,
                "author_id": inputs[i]["author_id"],
                "primary_ess_min": old[i]["qmc_ess_min"],
                "primary": old[i],
            }
            for power in (10, 12):
                refined = dict(bundle, qmc_power=power)
                a, b = (
                    panel([inputs[i]], refined, arm, ev),
                    panel([inputs[i]], refined, arm, ev, scramble=1),
                )
                row[f"nodes{2**power}"] = {
                    "scramble0": a[0],
                    "scramble1": b[0],
                    "scramble_difference": discrepancy(a, b),
                    "primary_difference": discrepancy([old[i]], a),
                }
            records.append(row)
            save(args.output / f"{seed}_{arm}.json", row)
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "arm": arm,
                        "primary_vs4096": row["nodes4096"]["primary_difference"],
                        "scrambles4096": row["nodes4096"]["scramble_difference"],
                    }
                ),
                flush=True,
            )
    save(
        args.output / "summary.json",
        {
            "cases": len(records),
            "unique_authors": len({(r["seed"], r["author_id"]) for r in records}),
            "max_primary_vs4096": {
                k: max(r["nodes4096"]["primary_difference"][k] for r in records)
                for k in ("score", "sd")
            },
            "max_scramble4096": {
                k: max(r["nodes4096"]["scramble_difference"][k] for r in records)
                for k in ("score", "sd")
            },
            "seconds": time.monotonic() - start,
            "scope": "only18ESSselected cases;no reestimated source/calibration or changed primary scores",
        },
    )


if __name__ == "__main__":
    main()
