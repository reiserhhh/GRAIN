#!/usr/bin/env python3
"""CON04: zero-neural-call evidence aggregation and graph-metric interpretation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import (
    BASE,
    digest,
    read_json,
    write_json,
)
from scripts.analyze_grain_conditional_navigation import NATURAL, PAIRS, ROUTES
from scripts.export_grain_conditional_navigation import DEFAULT as EXPORT
from scripts.fit_grain_conditional_navigation import DEFAULT as FIT
from scripts.fit_grain_conditional_navigation import response_table
from scripts.run_grain_conditional_augmentation import DEFAULT as AUG

DEFAULT = BASE / "conditional_navigation_v1/CON-04/closure_r1"


def graph_geometry(n=156, anchor_rows=60):
    """Four disjoint edit pairs per author leave108 response-invisible row directions."""
    if n <= anchor_rows or (n - anchor_rows) % 8:
        raise ValueError("Expected eight endpoints per author after the anchors")
    d = np.zeros(((n - anchor_rows) // 2, n))
    for i in range(len(d)):
        d[i, anchor_rows + 2 * i], d[i, anchor_rows + 2 * i + 1] = -1, 1
    return {
        "rows": n,
        "edges": len(d),
        "rank": int(np.linalg.matrix_rank(d)),
        "nullity": n - int(np.linalg.matrix_rank(d)),
        "nonzero_laplacian_eigenvalue": 2.0,
        "intercept_error": float(np.linalg.norm(d @ np.ones(n))),
    }


def main():
    """Record all candidates and paired uncertainty without changing any program."""
    out = DEFAULT
    if out.exists():
        raise ValueError("Closure evidence is immutable; use the existing deliverable")
    if read_json(EXPORT / "external/analysis/summary.json")["status"] != "COMPLETE":
        raise ValueError("Complete triggered external packet first")
    with threadpool_limits(limits=2):
        out.mkdir(parents=True)
        frames = []
        for kind, folder in (("PQ", FIT), ("PQ_plus_evidence", AUG / "fit")):
            for split in ("source", "natural"):
                x = pd.read_csv(folder / f"{split}_response_metrics.csv")
                x["input_kind"] = kind
                frames.append(x)
        pd.concat(frames, ignore_index=True).to_csv(
            out / "all_response_metrics.csv", index=False
        )
        worlds = read_json(NATURAL / "private_worlds.json")
        meta = pd.DataFrame(
            [{k: w[k] for k in ("world_id", "author_id", "condition")} for w in worlds]
        )
        path = ROUTES / "natural/predictions.csv"
        if digest(path) != read_json(path.parent / "prediction_freeze.json")["sha256"]:
            raise ValueError("Response comparator changed")
        route = (
            pd.read_csv(path)
            .query("reference=='R_synth' and method=='A_source'")
            .merge(meta, on="world_id", validate="many_to_one")
        )
        # response_table expects target fields; they are unused for baseline differences.
        route["split"], route["native"], route["fixed"] = "natural_exposed", 0.0, 0.0
        baseline = response_table(route)[
            ["query", "author_id", "axis", "score"]
        ].rename(columns={"score": "baseline"})
        response = pd.read_csv(AUG / "fit/private_natural_responses.csv").merge(
            baseline, on=["query", "author_id", "axis"], validate="many_to_one"
        )
        rows, rng = [], np.random.default_rng(42)
        for (q, method, axis), g in response.groupby(["query", "method", "axis"]):
            delta = (
                (g.score - g.native) ** 2 - (g.baseline - g.native) ** 2
            ).to_numpy()
            draws = delta[rng.integers(0, len(delta), size=(2000, len(delta)))].mean(1)
            lo, hi = np.quantile(draws, [0.025, 0.975])
            rows.append(
                {
                    "query": q,
                    "method": method,
                    "axis": axis,
                    "n": len(delta),
                    "delta_mse": float(delta.mean()),
                    "ci_lower": float(lo),
                    "ci_upper": float(hi),
                    "scope": "author-paired exposed panel, conditional on frozen fits, no multiplicity correction",
                }
            )
        pd.DataFrame(rows).to_csv(
            out / "augmentation_paired_uncertainty.csv", index=False
        )
        with np.load(
            AUG / "fit/private_inputs_extraversion.npz", allow_pickle=False
        ) as z:
            x = z["x"]
        pq, sm = x[:, :-3072], x[:, -3072:]
        pq, sm = pq - pq.mean(0), sm - sm.mean(0)
        k, s = pq @ pq.T, sm @ sm.T
        geometry = {
            "summary_fraction_centered_energy": float(np.trace(s) / np.trace(k + s)),
            "summary_vs_PQ_gram_frobenius_ratio": float(
                np.linalg.norm(s) / np.linalg.norm(k)
            ),
            "augmented_vs_PQ_gram_element_r": float(
                np.corrcoef(k.ravel(), (k + s).ravel())[0, 1]
            ),
        }
        # Same disjoint-pair spectrum regardless of endpoint storage order.
        graph = graph_geometry()
        diagnostics = pd.read_csv(FIT / "construction_diagnostics.csv")
        diagnostics["pair_direction_row_metric_eigenvalue"] = (
            1 + 2 * diagnostics["lambda"] * 156 / 48
        )
        diagnostics.to_csv(out / "objective_row_metric.csv", index=False)
        if len(PAIRS) != 4:
            raise ValueError("Declared graph changed")
        paths = [
            FIT / "construction_freeze.json",
            FIT / "prediction_freeze.json",
            AUG / "collection_freeze.json",
            AUG / "fit/construction_freeze.json",
            AUG / "fit/prediction_freeze.json",
            EXPORT / "selection_freeze.json",
            EXPORT / "replay_conditional/summary.json",
            EXPORT / "external/prediction_freeze.json",
            EXPORT / "external/analysis/summary.json",
        ]
        write_json(
            out / "summary.json",
            {
                "status": "COMPLETE_CON00_TO_CON04_FINITE_DELIVERY",
                "new_neural_calls": 0,
                "augmentation_geometry": geometry,
                "response_graph": graph,
                "artifacts": {str(p.relative_to(ROOT)): digest(p) for p in paths},
                "code_sha256": digest(Path(__file__)),
                "decision": "retain conditional TF/state operators as response models; retain B0 TF/B4 state as external-link anchors; E augmentation stopped",
                "next_mechanism": "join within-text conditional responses with between-author reference positioning without human-label tuning",
                "universal_theory_complete": False,
                "human_labels_fit": False,
            },
        )
    print(read_json(out / "summary.json"), flush=True)


if __name__ == "__main__":
    main()
