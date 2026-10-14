#!/usr/bin/env python3
"""Replay M01/M09 bindings and quantify parser-induced evidence selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.proposition_response import (
    bind_reply,
    condition_text,
    frames,
    response_coordinates,
)
from grain_research.proposition_scope_accounting import mean_bounds, role_account
from scripts.run_grain_explanation_chain import sha, write
from scripts.run_grain_proposition_response import BASE


def audit(out):
    """Check saved parser words/addresses, independent means and finite bounds."""
    dst = out / "audit"
    dst.mkdir(exist_ok=False)
    frozen = json.loads((out / "freeze.json").read_text())
    for path, digest in frozen["files"].items():
        assert sha(ROOT / path) == digest, path
    completion = out / "completion"
    manifest = json.loads((completion / "manifest.json").read_text())
    for path, digest in manifest.items():
        assert sha(completion / path) == digest, path
    contract = json.loads((completion / "resume_contract.json").read_text())
    for path, digest in contract["files"].items():
        assert sha(out / path) == digest, path
    raw = json.loads((out / "private_inputs.json").read_text())
    graphs = {x["id"]: x for x in json.loads((out / "private_graphs.json").read_text())}
    parse = {
        x["text"]: x["sentences"]
        for x in json.loads((out / "private_parses.json").read_text())
    }
    corrected, addresses = [], 0
    for row in raw:
        partner, reply = row["partner"], row["reply"]
        pf, rf = (
            frames(partner, parse[partner], "partner"),
            frames(reply, parse[reply], "respondent"),
        )
        old = graphs[row["id"]]
        assert pf == old["partner_frames"] and rf == old["reply_frames"]
        binding = bind_reply(partner, reply, pf, rf)
        assert binding == old["binding"]
        coordinate, supported = response_coordinates(binding)
        assert (
            coordinate == old["coordinates"]
            and condition_text(binding) == old["condition_text"]
        )
        for fs, text in [(pf, partner), (rf, reply)]:
            for f in fs:
                assert text[f["start"] : f["end"]] == f["quote"]
                addresses += 1
                for c in f["conditions"]:
                    assert text[c["start"] : c["end"]] == c["text"]
                    addresses += 1
                corrected.append(
                    {
                        "record_id": row["id"],
                        "part": row["part"],
                        **role_account(f, explicit_addressee=bool(partner)),
                    }
                )
    write(dst / "private_role_accounting.json", corrected)
    table = pd.read_csv(completion / "controlled_readings.csv").set_index("id")
    ledger = pd.read_csv(completion / "simulated_event_ledger.csv")
    theta = np.load(completion / "simulated_parameters.npy", allow_pickle=False)
    predictions = pd.read_csv(completion / "simulated_profiles.csv")
    replay, intervals, bias = [], [], []
    for (person, wording), events in ledger.groupby(["person", "wording"]):
        estimates = table.loc[events["case"]].reset_index(drop=True)
        true_event = events.oracle.to_numpy()
        real_condition = events.condition.to_numpy()
        estimated_condition = estimates.estimated_condition.to_numpy()
        graph = estimates.graph_target.to_numpy()
        for c in [0, 1]:
            truth = 0.7 * (theta[person, c, 0] - theta[person, c, 1]) / 2
            for arm, assigned, scores in [
                ("oracle", real_condition, true_event),
                ("graph_oracle_condition", real_condition, graph),
                ("graph_estimated_condition", estimated_condition, graph),
                ("content", real_condition, estimates.content_target.to_numpy()),
            ]:
                mask = assigned == c
                usable = mask & np.isfinite(scores)
                value = (
                    float(scores[usable].sum() / usable.sum()) if usable.any() else None
                )
                replay.append(
                    {
                        "person": person,
                        "wording": wording,
                        "arm": arm,
                        "condition": c,
                        "score": value,
                        "truth": truth,
                        "opportunities": int(mask.sum()),
                        "observed": int(usable.sum()),
                    }
                )
            mask = real_condition == c
            valid = mask & np.isfinite(graph)
            n, m = int(valid.sum()), int((mask & ~np.isfinite(graph)).sum())
            total = float(graph[valid].sum())
            finite_truth = float(true_event[mask].mean())
            bound = mean_bounds(total, n, m)
            assert bound["lower"] - 1e-12 <= finite_truth <= bound["upper"] + 1e-12
            unknown = int((~np.isfinite(estimated_condition)).sum())
            known = (estimated_condition == c) & np.isfinite(graph)
            unknown_bound = mean_bounds(
                float(graph[known].sum()), int(known.sum()), unknown
            )
            assert (
                unknown_bound["lower"] - 1e-12
                <= finite_truth
                <= unknown_bound["upper"] + 1e-12
            )
            intervals.append(
                {
                    "person": person,
                    "wording": wording,
                    "condition": c,
                    "finite_truth": finite_truth,
                    "n": n,
                    "missing": m,
                    "known_lower": bound["lower"],
                    "known_upper": bound["upper"],
                    "unknown_condition_lower": unknown_bound["lower"],
                    "unknown_condition_upper": unknown_bound["upper"],
                }
            )
            if m:
                selection = np.isfinite(graph[mask]).astype(float)
                z = true_event[mask]
                cov = np.mean((selection - selection.mean()) * (z - z.mean()))
                measured = float(graph[valid].mean())
                assert abs(measured - finite_truth - cov / selection.mean()) < 1e-12
                bias.append(
                    {
                        "person": person,
                        "wording": wording,
                        "condition": c,
                        "observed_mean": measured,
                        "finite_mean": finite_truth,
                        "coverage": selection.mean(),
                        "bias": measured - finite_truth,
                        "selection_covariance": cov,
                    }
                )
    rebuilt = pd.DataFrame(replay)
    keys = ["person", "wording", "arm", "condition"]
    pd.testing.assert_frame_equal(
        predictions.set_index(keys).sort_index()[rebuilt.columns.difference(keys)],
        rebuilt.set_index(keys).sort_index()[rebuilt.columns.difference(keys)],
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
    )
    metrics = json.loads((completion / "simulation_metrics.json").read_text())
    for item in metrics:
        rows = rebuilt.loc[
            rebuilt.wording.eq(item["wording"]) & rebuilt.arm.eq(item["arm"])
        ].dropna(subset=["score"])
        mse = float(np.mean((rows.score - rows.truth) ** 2))
        assert abs(mse - item["mse"]) < 1e-12
        assert abs(float(np.corrcoef(rows.score, rows.truth)[0, 1]) - item["r"]) < 1e-12
    bounds = pd.DataFrame(intervals)
    bounds.to_csv(dst / "finite_material_bounds.csv", index=False)
    pd.DataFrame(bias).to_csv(dst / "selection_bias.csv", index=False)
    held = bounds[bounds.wording.eq(1)]
    correction_count = sum(
        x["proposition_actor_candidate"] == "second_person_unresolved"
        for x in corrected
    )
    write(
        dst / "summary.json",
        {
            "status": "pass",
            "neural_calls": 0,
            "frozen_files": len(frozen["files"]),
            "raw_graphs": len(raw),
            "predicate_condition_addresses": addresses,
            "replayed_profiles": len(rebuilt),
            "replayed_metrics": len(metrics),
            "finite_bounds": len(bounds),
            "selection_identities": len(bias),
            "known_condition_held_mean_width": float(
                (held.known_upper - held.known_lower).mean()
            ),
            "unknown_condition_held_mean_width": float(
                (held.unknown_condition_upper - held.unknown_condition_lower).mean()
            ),
            "standalone_second_person_scope_corrections": correction_count,
            "scope_correction_changes_old_scores": False,
            "bound_is_population_CI": False,
            "post_result_analysis": True,
            "independent_executor": False,
            "script_sha256": sha(__file__),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=BASE / "proposition_response_r1")
    audit(parser.parse_args().out)
