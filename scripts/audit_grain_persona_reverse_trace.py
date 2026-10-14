#!/usr/bin/env python3
"""Independent artifact arithmetic, selection, and prefix/readout audit (no model)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_grain_persona_generation_trace import BASE, read_jsonl, sha, write


def address_paths(directory):
    """Exclude macOS AppleDouble metadata, not malformed scientific JSON files."""
    return sorted(p for p in directory.glob("*.json") if not p.name.startswith("._"))


def audit(run, endpoint, full_context):
    """Recompute scores/selection independently and relate frozen GRAIN coordinates."""
    rows = read_jsonl(run / "effects.jsonl")
    if json.loads((run / "status.json").read_text())["status"] != "COMPLETE":
        raise ValueError("Reverse scan is not complete")
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["source_sha256"] == sha(BASE / "expanded/generations.jsonl")
    assert len(rows) == len({r["key"] for r in rows})
    baselines = {
        (r["pair"], r["query"], r["site"], r["kind"]): r
        for r in rows
        if r["stage"] == "baseline"
    }
    errors, self_effects = [], []
    for row in rows:
        low = baselines[(row["pair"], row["query"], "prefix", "low")]["score"]
        high = baselines[(row["pair"], row["query"], "prefix", "high")]["score"]
        errors.extend(
            [
                abs(row["effect"] - (row["score"] - low)),
                abs(row["contrast"] - (high - low)),
            ]
        )
        assert 0 <= row["score"] <= 1 and 0 <= row["mass"] <= 1
        if row["kind"] == "self":
            self_effects.append(abs(row["effect"]))
    selection = {}
    for stage, filename in [
        ("residual", "selected_residual.json"),
        ("heads", "selected_head.json"),
    ]:
        groups = {}
        for row in rows:
            if (
                row["stage"] == stage
                and row["kind"] == "donor"
                and row["role"] == "discovery"
                and row["axis"] == row["query"] == "extraversion"
                and ".all" not in row["site"]
            ):
                groups.setdefault(row["site"], []).append(
                    (row["effect"] - row["contrast"]) ** 2
                )
        chosen = min(groups, key=lambda k: (sum(groups[k]) / len(groups[k]), str(k)))
        assert chosen == json.loads((run / filename).read_text())["site"]
        selection[stage] = chosen
    address_checks = 0
    old = json.loads((BASE / "expanded/readout_local_query.json").read_text())["raw"]
    old_error = []
    grain = json.loads((BASE / "expanded/readout_grain.json").read_text())["grain"]
    grain_delta = {}
    for path in address_paths(run / "addresses"):
        a = json.loads(path.read_text())
        pos, h, l = a["article_positions"], a["high_prompt_ids"], a["low_prompt_ids"]
        assert (
            len(h) == len(l)
            and h[: pos[0]] == l[: pos[0]]
            and h[pos[-1] + 1 :] == l[pos[-1] + 1 :]
        )
        address_checks += 1
        for pole in ("high", "low"):
            value = baselines[(a["pair"], a["query"], "full", pole)]["score"]
            old_error.append(abs(value - old[a[pole + "_id"]][a["query"]]))
        grain_delta[a["pair"]] = (
            np.asarray(grain[a["high_id"]]["g"]) - grain[a["low_id"]]["g"]
        )
    readback = {}
    for axis in ("extraversion", "neuroticism"):
        for query in ("extraversion", "neuroticism"):
            for task in ("opinion", "advice", "neighbor"):
                for context in ("full", "prefix"):
                    subset = [
                        r
                        for r in rows
                        if r["stage"] == "baseline"
                        and r["site"] == context
                        and r["axis"] == axis
                        and r["query"] == query
                        and r["situation"] == task
                    ]
                    if not subset:
                        continue
                    high = [r["score"] for r in subset if r["kind"] == "high"]
                    low = [r["score"] for r in subset if r["kind"] == "low"]
                    readback[f"{axis}|{query}|{task}|{context}"] = {
                        "n_pairs": len(high),
                        "auc": roc_auc_score(
                            [1] * len(high) + [0] * len(low), high + low
                        ),
                        "mean_contrast": float(np.mean(high) - np.mean(low)),
                    }
    associations = {}
    for site in (selection["heads"], "L16.H7", "L20.H5"):
        subset = [
            r
            for r in rows
            if r["stage"] == "heads" and r["site"] == site and r["kind"] == "donor"
        ]
        x = np.stack([grain_delta[r["pair"]] for r in subset])
        y = np.asarray([r["effect"] for r in subset])
        associations[site] = {
            "n": len(subset),
            "r_by_grain_coordinate": [
                float(pearsonr(x[:, i], y).statistic) for i in range(x.shape[1])
            ],
            "scope": "exploratory full-article GRAIN delta vs prefix-reader effect; not validation",
        }
    endpoint_checks = 0
    if (endpoint / "effects.jsonl").exists():
        assert (
            json.loads((endpoint / "status.json").read_text())["status"] == "COMPLETE"
        )
        for row in read_jsonl(endpoint / "effects.jsonl"):
            errors.extend(
                [
                    abs(row["effect"] - row["before"]["score"] + row["after"]["score"]),
                    abs(row["contrast"] - row["effect"] - row["remaining"]),
                ]
            )
            if row["kind"] == "self":
                self_effects.append(abs(row["effect"]))
            endpoint_checks += 1
    full_checks = 0
    full_endpoint_checks = 0
    full_associations = {}
    if (full_context / "effects.jsonl").exists():
        assert (
            json.loads((full_context / "status.json").read_text())["status"]
            == "COMPLETE"
        )
        full_rows = read_jsonl(full_context / "effects.jsonl")
        assert len(full_rows) == len({r["key"] for r in full_rows})
        for row in full_rows:
            errors.extend(
                [
                    abs(row["effect"] - row["result"]["score"] + row["low"]["score"]),
                    abs(row["contrast"] - row["high"]["score"] + row["low"]["score"]),
                ]
            )
            assert (
                len(row["high_suffix"])
                == len(row["low_suffix"])
                == len(row["suffix_ids"])
            )
            if row["kind"] == "self":
                self_effects.append(abs(row["effect"]))
            full_checks += 1
        for site in (selection["heads"], "L16.H7", "L20.H5"):
            subset = [
                r for r in full_rows if r["site"] == site and r["kind"] == "donor"
            ]
            x = np.stack([grain_delta[r["pair"]] for r in subset])
            y = np.asarray([r["effect"] for r in subset])
            full_associations[site] = {
                "n": len(subset),
                "r_by_grain_coordinate": [
                    float(pearsonr(x[:, i], y).statistic) for i in range(x.shape[1])
                ],
                "scope": "post-result descriptive association; same full articles; no psychological criterion",
            }
    full_endpoint = run.parent / "full_endpoint_r1"
    if (full_endpoint / "effects.jsonl").exists():
        assert (
            json.loads((full_endpoint / "status.json").read_text())["status"]
            == "COMPLETE"
        )
        endpoint_old = json.loads(
            (BASE / "pilot/endpoint_local_query.json").read_text()
        )["raw"]
        e_rows = read_jsonl(full_endpoint / "effects.jsonl")
        assert len(e_rows) == len({r["key"] for r in e_rows})
        for row in e_rows:
            errors.extend(
                [
                    abs(row["effect"] - row["before"]["score"] + row["after"]["score"]),
                    abs(row["contrast"] - row["effect"] - row["remaining"]),
                ]
            )
            prefix = row["pair"]
            seed = row["seed"]
            assert (
                abs(
                    row["before"]["score"]
                    - endpoint_old[f"{prefix}|{row['arm']}|s{seed}"]["extraversion"]
                )
                < 1e-6
            )
            assert (
                abs(
                    row["low"]["score"]
                    - endpoint_old[f"{prefix}|control_low|s{seed}"]["extraversion"]
                )
                < 1e-6
            )
            if row["kind"] == "self":
                self_effects.append(abs(row["effect"]))
            full_endpoint_checks += 1
    assert max(errors) < 1e-10 and max(self_effects) < 1e-6
    report = {
        "status": "PASS",
        "rows": len(rows),
        "address_checks": address_checks,
        "endpoint_checks": endpoint_checks,
        "full_context_checks": full_checks,
        "full_endpoint_checks": full_endpoint_checks,
        "max_arithmetic_error": max(errors),
        "max_self_effect": max(self_effects),
        "old_full_readout_max_error": max(old_error),
        "selection": selection,
        "readback": readback,
        "grain_exploratory": associations,
        "grain_full_context_exploratory": full_associations,
        "note": "Independent arithmetic implementation by same agent, not independent third-party review.",
    }
    assert max(old_error) < 1e-6
    write(run / "audit.json", report)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                not in (
                    "readback",
                    "grain_exploratory",
                    "grain_full_context_exploratory",
                )
            },
            indent=2,
        )
    )


def main():
    """CLI for frozen reverse-run and optional endpoint checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        default=ROOT / "results/grain_vector_mechanisms/persona_reverse_trace_v2/r1",
    )
    parser.add_argument(
        "--endpoint",
        type=Path,
        default=ROOT
        / "results/grain_vector_mechanisms/persona_reverse_trace_v2/endpoint_r1",
    )
    parser.add_argument(
        "--full-context",
        type=Path,
        default=ROOT
        / "results/grain_vector_mechanisms/persona_reverse_trace_v2/full_context_r1",
    )
    args = parser.parse_args()
    audit(args.run, args.endpoint, args.full_context)


if __name__ == "__main__":
    main()
