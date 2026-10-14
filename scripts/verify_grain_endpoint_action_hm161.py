#!/usr/bin/env python3
"""Reconstruct endpoint probabilities, actual source donors and finite actions."""

from __future__ import annotations

import csv
import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_endpoint_action_hm161 as E


def verify(out=E.DEFAULT):
    """Numerically audit the full finite support, without new decoder forwards."""
    start, m = time.monotonic(), E.check(out)
    R, C = E.R, E.C
    E.G.authenticate_results(out)
    rows, queries, design = (
        R.read(out / f"{name}.json") for name in ("rows", "queries", "design")
    )
    if len(design) != 50 or len(queries["definitions"]) != 16:
        raise ValueError("Incomplete registered endpoint support")
    parent = ROOT / m["parent"]
    old = E.parent_natural(parent)
    source = R.arrays(parent / "source_operations.npz")
    weights = (
        R.OLD.weights()
        .cpu()[[queries["answers"][s][0] for s in ("Yes", "No")]]
        .numpy()
        .astype(float)
    )
    raw, scores = {}, {}
    donor_error = head_error = probability_error = character_error = 0.0
    prefix_error = self_error = replay_error = 0.0
    original_queries = R.read(parent / "queries.json")["definitions"]
    for entry in design:
        if time.monotonic() - start > m["verification_seconds_cap"]:
            raise TimeoutError("Finite verification cap; no automatic extension")
        row, mode = rows[entry["row"]], entry["mode"]
        a = R.arrays(out / "captures" / f"{entry['batch']:03d}.npz")
        raw[row["id"], mode] = a
        expected_head = a["states"].astype(float) @ weights.T
        head_error = max(
            head_error, float(np.max(abs(expected_head - a["selected_logits"])))
        )
        if not np.allclose(expected_head, a["selected_logits"], atol=1e-4, rtol=1e-5):
            raise ValueError("CPU64 native output head reconstruction failed")
        if a["blocks"].tolist() != list(range(28)) or float(a["cache_spread"]) > 1e-4:
            raise ValueError("Historical causal support/block execution changed")
        score = {key: [] for key in E.M.MEASURES}
        for logits, norm in zip(a["selected_logits"], a["normalizer"], strict=True):
            y, n, z = float(logits[0]), float(logits[1]), float(norm)
            py, pn = math.exp(y - z), math.exp(n - z)
            score["log_odds"].append(y - n)
            score["restricted_yes"].append(py / (py + pn))
            score["full_yes"].append(py)
            score["answer_mass"].append(py + pn)
        scores[row["id"], mode] = score
        base = old[row["id"]] if mode == "natural" else raw[row["id"], "natural"]
        prefix_error = max(
            prefix_error,
            float(np.max(abs(a["prefix_states"] - base["prefix_states"]))),
        )
        if mode == "natural":
            for i, query in enumerate(queries["definitions"]):
                if query["family"] == "diagnostic":
                    j = original_queries.index({**query, "family": "named"})
                    replay_error = max(
                        replay_error,
                        float(
                            np.max(
                                abs(
                                    a["selected_logits"][i] - base["selected_logits"][j]
                                )
                            )
                        ),
                    )
            continue
        slots = (
            list(range(len(row["history"])))
            if mode == "self"
            else row["scopes"]["neuroticism" if mode[0] == "e" else "extraversion"]
        )
        if a["patched_slots"].tolist() != slots:
            raise ValueError("Actual intervention support changed")
        for channel in ("key", "value"):
            name = "history_" + channel
            expected = base[name].copy()
            if mode != "self":
                cube = source[f"{row['condition']}_{row['order']}_{name}"].astype(float)
                mixed = (
                    sum(
                        ei * ni * cube[(ei + 1) // 2, (ni + 1) // 2]
                        for ei, ni in itertools.product((-1, 1), repeat=2)
                    )
                    / 4
                )
                expected[:, :, slots] = (
                    expected[:, :, slots].astype(float)
                    - 2 * row["e"] * row["n"] * mixed[:, :, slots]
                ).astype(expected.dtype)
            error = float(
                np.max(
                    abs(expected.astype(float) - a["donor_" + channel].astype(float))
                )
            )
            donor_error = max(donor_error, error)
        if mode == "self":
            self_error = max(
                self_error,
                float(np.max(abs(a["selected_logits"] - base["selected_logits"]))),
            )
    if donor_error > 1e-6 or max(prefix_error, self_error, replay_error) > 1e-4:
        raise ValueError("Source construction, replay or causal-prefix check failed")
    with (out / "endpoints.csv").open() as f:
        endpoints = list(csv.DictReader(f))
    if len(endpoints) != 800:
        raise ValueError("Missing endpoint observations")
    for record in endpoints:
        row, mode = int(record["row"]), record["mode"]
        qi = queries["definitions"].index(
            {
                "axis": record["axis"],
                "family": record["family"],
                "person": int(record["person"]),
            }
        )
        for measure in E.M.MEASURES:
            probability_error = max(
                probability_error,
                abs(float(record[measure]) - scores[row, mode][measure][qi]),
            )
    indexed = {(r["condition"], r["order"], r["e"], r["n"]): r["id"] for r in rows}
    with (out / "characters.csv").open() as f:
        chars = list(csv.DictReader(f))
    if len(chars) != 6144:
        raise ValueError("Missing complete-character actions")
    for record in chars:
        condition, order, mask = (
            int(record[k]) for k in ("condition", "order", "mask")
        )
        axis, component, measure = (record[k] for k in ("axis", "component", "measure"))
        terms = []
        for e, n, p in itertools.product((-1, 1), repeat=3):
            row = indexed[condition, order, e, n]
            qi = queries["definitions"].index(
                {"axis": record["target"], "family": record["family"], "person": p}
            )
            baseline = scores[row, "natural"][measure][qi]
            if component == "base":
                value = baseline
            elif component == "source":
                value = scores[row, axis + "_relation"][measure][qi] - baseline
            else:
                partner = indexed[
                    condition, order, -e if axis == "e" else e, -n if axis == "n" else n
                ]
                value = scores[partner, "natural"][measure][qi] - baseline
            weight = math.prod(
                s for bit, s in enumerate((e, n, p)) if mask & (1 << bit)
            )
            terms.append(weight * value)
        character_error = max(
            character_error, abs(math.fsum(terms) / 8 - float(record["coefficient"]))
        )
    if max(character_error, probability_error) > 1e-10:
        raise ValueError("Alternate endpoint or character arithmetic failed")
    report = {
        "status": "NUMERICAL_AUDIT_PASS_NOT_PSYCHOLOGICAL_VALIDITY",
        "actual_batches": len(design),
        "endpoint_rows": len(endpoints),
        "character_rows": len(chars),
        "max_donor_error": donor_error,
        "max_cpu64_head_error": head_error,
        "max_endpoint_error": probability_error,
        "max_character_error": character_error,
        "max_prefix_error": prefix_error,
        "max_self_error": self_error,
        "max_old_diagnostic_replay_error": replay_error,
        "same_agent_alternate_arithmetic_not_blind": True,
        "seconds": time.monotonic() - start,
    }
    if not (out / "alternate_audit.json").exists():
        C.P.write_json(out / "alternate_audit.json", report)
    print(report, flush=True)
    return report


if __name__ == "__main__":
    verify()
