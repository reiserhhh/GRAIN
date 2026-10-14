#!/usr/bin/env python3
"""Alternate enumerative audit of the source construction and cold target operator."""

from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_ternary_meaning_hm180 as U


def verify(out):
    """Recompute actual author triples, profile margins and cold sums independently."""
    R.check(out, ("compiled_seal.json", "scored_seal.json", "analysis_seal.json"))
    z = R.prepared(out)
    max_full = max_cold = max_constraint = max_account = max_loss = 0.0
    count = cold_values = crossfit_count = 0
    with (out / "source_crossfit_authors.csv").open() as stream:
        losses = {
            (int(r["author"]), int(r["view"]), r["target"]): r
            for r in csv.DictReader(stream)
        }
    for held in (-1, 0, 1):
        path = "source_bundle.json" if held == -1 else f"crossfit_bundle_{held}.json"
        b = R.read(out / path)
        profiles, tables = (
            np.asarray(b["profiles"], int),
            np.asarray(b["profile_tables"], float),
        )
        group = {tuple(row): i for i, row in enumerate(profiles)}
        used = np.arange(386) if held == -1 else np.flatnonzero(z["folds"] != held)
        direct = np.zeros((2, 32, 8, 8, 8))
        for u in used:
            g = group[tuple(z["source_targets"][u])]
            for v in range(2):
                c = z["source_cells"][u, v]
                for i, j, k in itertools.product(range(4), range(4, 8), range(8, 12)):
                    direct[v, g, c[i], c[j], c[k]] += 1 / (len(used) * 64)
                    count += 1
        max_full = max(max_full, float(np.max(np.abs(direct - tables[:, 1]))))
        for v in range(2):
            for axis in range(4):
                max_constraint = max(
                    max_constraint,
                    float(
                        np.max(np.abs(tables[v, 0].sum(axis) - tables[v, 1].sum(axis)))
                    ),
                )
            for q in range(5):
                positive = np.stack(
                    [tables[v, m, profiles[:, q] == 1].sum(0) for m in range(2)]
                )
                negative = np.stack(
                    [tables[v, m, profiles[:, q] == 0].sum(0) for m in range(2)]
                )
                total = positive + negative
                field = np.divide(
                    positive, total, out=np.full_like(total, np.nan), where=total > 0
                )
                dpos, dtotal = positive[1] - positive[0], total[1] - total[0]
                good = np.isfinite(field).all(0)
                account = (dpos[good] - field[0][good] * dtotal[good]) / total[1][good]
                max_account = max(
                    max_account,
                    float(np.max(np.abs(account - (field[1] - field[0])[good]))),
                )
        if held == -1:
            main_b = b
        else:
            for u in np.flatnonzero(z["folds"] == held):
                for v in range(2):
                    c = z["source_cells"][u, v]
                    totals, known = np.zeros((2, 5)), 0
                    for i, j, k in itertools.product(
                        range(4), range(4, 8), range(8, 12)
                    ):
                        vectors = tables[v, :, :, c[i], c[j], c[k]]
                        den = vectors.sum(1)
                        crossfit_count += 1
                        if np.any(den == 0):
                            continue
                        ps = vectors @ profiles / den[:, None]
                        totals += (ps - z["source_targets"][u]) ** 2
                        known += 1
                    for q, target in enumerate(U.T.TARGETS):
                        row = losses[(int(z["source"][u]), v, target)]
                        if (
                            int(row["supported_triples"]) != known
                            or int(row["held_fold"]) != held
                        ):
                            raise ValueError("Held-author identity/support changed")
                        if known:
                            got = np.array(
                                [
                                    float(row["reference_brier_on_common_support"]),
                                    float(row["full_brier_on_common_support"]),
                                ]
                            )
                            max_loss = max(
                                max_loss,
                                float(np.max(np.abs(got - totals[:, q] / known))),
                            )
    table, profiles = (
        np.asarray(main_b["profile_tables"], float),
        np.asarray(main_b["profiles"], float),
    )
    outputs = {
        r["author_key"]: r
        for r in map(json.loads, (out / "cold_output.jsonl").read_text().splitlines())
    }
    for record in map(json.loads, (out / "cold_input.jsonl").read_text().splitlines()):
        roles = [[e for e in record["events"] if e["slot"] // 4 == r] for r in range(3)]
        total, known, n = np.zeros((2, 32)), True, 0
        for a, b, c in itertools.product(*roles):
            vector = table[record["view"], :, :, a["cell"], b["cell"], c["cell"]]
            den = vector.sum(1)
            n += 1
            if np.any(den == 0):
                known = False
            else:
                total += vector / den[:, None]
        expected = total / 64 if known else np.full_like(total, np.nan)
        got = outputs[record["author_key"]]
        original = np.asarray(got["profile_mass_readout"], float)
        if (
            not np.array_equal(np.isfinite(expected), np.isfinite(original))
            or got["triples"] != n
        ):
            raise ValueError("Cold missing/opportunity account changed")
        mask = np.isfinite(expected)
        if mask.any():
            max_cold = max(
                max_cold, float(np.max(np.abs(expected[mask] - original[mask])))
            )
        target, got_target = (
            expected @ profiles,
            np.asarray(got["target_mass_readout"], float),
        )
        mask = np.isfinite(target)
        if mask.any():
            max_cold = max(
                max_cold, float(np.max(np.abs(target[mask] - got_target[mask])))
            )
        cold_values += target.size
        if n and known and np.max(np.abs((expected * 64 / n).sum(1) - 1)) > 1e-12:
            raise ValueError("Cold joint-profile coherence failed")
    if (
        max_full > 2e-12
        or max_cold > 2e-12
        or max_constraint > 8e-10
        or max_account > 2e-12
        or max_loss > 2e-12
    ):
        raise ValueError("Fixed construction arithmetic verification failed")
    result = {
        "enumerated_source_triples": count,
        "crossfit_triples": crossfit_count,
        "crossfit_rows": len(losses),
        "cold_records": len(outputs),
        "cold_target_values": cold_values,
        "full_law_error": max_full,
        "constraint_error": max_constraint,
        "target_account_error": max_account,
        "cold_coherence_error": max_cold,
        "crossfit_loss_error": max_loss,
        "same_agent_not_blind": True,
        "status": "PASS",
    }
    if (out / "verification.json").exists():
        if R.read(out / "verification.json") != result:
            raise ValueError("Read-only verification disagrees with previous audit")
    else:
        R.write(out / "verification.json", result)
        R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(result))


if __name__ == "__main__":
    verify(Path(sys.argv[1]) if len(sys.argv) > 1 else U.DEFAULT)
