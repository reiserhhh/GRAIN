#!/usr/bin/env python3
"""Decompose the F08 end-to-end reader-fitted source shift (tables only).

Re-renders the exact source readings of `F08/end_to_end_r1` (same streams) and
refits the gap-aware B1 table under four arms: oracle (all true values), the
reader's missingness pattern with true values (selection only), the reader's
values with no missingness (value errors only), and the full reader read. No
application scoring and no new reader forwards.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from run_grain_gptpro_f08_end_to_end import COUNTS, READERS, SEEDS, render, runs

from grain_research.gptpro_foundation_v3.f02 import generate_world
from grain_research.measurement_discovery.estimators import ordered_rows


def b1_table(histories):
    """Pooled gap-aware transition table P(next=1 | previous two)."""
    counts = np.zeros((4, 2))
    for rows in histories:
        for seg in runs(rows):
            for i in range(2, len(seg)):
                counts[2 * seg[i - 2] + seg[i - 1], seg[i]] += 1
    return (counts[:, 1] + 1) / (counts.sum(axis=1) + 2), counts.sum(axis=1)


def main():
    """Four source arms per reader x world x seed on the mixed panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    out = []
    for reader, folder in READERS.items():
        with (folder / "reader_scores.csv").open() as f:
            cache = {
                (r["role"], int(r["pole"]), r["subject_status"], int(r["variant"])): r
                for r in csv.DictReader(f)
            }
        for world_index, world in enumerate(("c2_xor", "c2_iid")):
            for seed in SEEDS:
                records, _ = generate_world(world, seed, counts=COUNTS)
                source = [r for r in records if r["role"] == "source"]
                arms = {k: [] for k in ("oracle", "selection_only", "values_only", "full_read")}
                selected_pos = selected_n = 0
                for s_index, rows in enumerate(ordered_rows(source).values()):
                    ys = [int(r["observation"]) for r in rows]
                    rd = render(
                        np.random.SeedSequence([seed, 8, 100 + world_index, s_index]),
                        "mixed",
                        ys,
                        cache,
                        "source",
                    )
                    keep = [x["hard"][1] >= 0.5 for x in rd]
                    value = [int(x["hard"][0]) for x in rd]
                    arms["oracle"].append(ys)
                    arms["selection_only"].append(
                        [y if k else None for y, k in zip(ys, keep, strict=True)]
                    )
                    arms["values_only"].append(
                        [v if k else y for y, v, k in zip(ys, value, keep, strict=True)]
                    )
                    arms["full_read"].append(
                        [v if k else None for v, k in zip(value, keep, strict=True)]
                    )
                    selected_pos += sum(y for y, k in zip(ys, keep, strict=True) if k)
                    selected_n += sum(keep)
                oracle_p, _ = b1_table(arms["oracle"])
                for arm, hist in arms.items():
                    p, n = b1_table(hist)
                    out.append(
                        {
                            "reader": reader,
                            "world": world,
                            "seed": seed,
                            "arm": arm,
                            "p00": p[0],
                            "p01": p[1],
                            "p10": p[2],
                            "p11": p[3],
                            "triples": int(n.sum()),
                            "max_abs_shift_vs_oracle": float(np.abs(p - oracle_p).max()),
                            "true_positive_rate_among_read": selected_pos / selected_n,
                            "true_positive_rate_all": float(
                                np.mean([y for h in arms["oracle"] for y in h])
                            ),
                        }
                    )
    with (args.output / "source_table_decomposition.csv").open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    summary = {}
    for reader in READERS:
        for arm in ("oracle", "selection_only", "values_only", "full_read"):
            vals = [r["max_abs_shift_vs_oracle"] for r in out if r["reader"] == reader and r["arm"] == arm]
            summary[f"{reader}:{arm}"] = {"mean_max_shift": float(np.mean(vals)), "max": float(np.max(vals))}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
