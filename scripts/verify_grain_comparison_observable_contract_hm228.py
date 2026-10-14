#!/usr/bin/env python3
"""Independently verify global ordinal feasibility and retain cycle witnesses."""

import argparse
import sys
from collections import deque
from itertools import pairwise
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_author_comparison_geometry_hm226 as G
from scripts import run_grain_comparison_observable_contract_hm228 as C
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_prospective_evidence_hm218 as H


def topological_or_cycle(adjacency):
    """Use independent Kahn traversal/DFS, not the primary SCC implementation."""
    n = len(adjacency)
    indegree = adjacency.sum(axis=0).astype(int)
    queue = deque(np.flatnonzero(indegree == 0).tolist())
    ordered = []
    while queue:
        u = queue.popleft()
        ordered.append(int(u))
        for v in np.flatnonzero(adjacency[u]):
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(int(v))
    if len(ordered) == n:
        return ordered, []
    visited, active, path = set(), set(), []

    def visit(u):
        visited.add(u)
        active.add(u)
        path.append(u)
        for v in np.flatnonzero(adjacency[u]).tolist():
            if v in active:
                return path[path.index(v) :] + [v]
            if v not in visited:
                cycle = visit(v)
                if cycle:
                    return cycle
        active.remove(u)
        path.pop()
        return []

    for u in range(n):
        if u not in visited:
            cycle = visit(u)
            if cycle:
                return [], cycle
    raise ValueError("Kahn/DFS disagreement")


def verify(out):
    """Verify all ten ordinal accounts and sensitivity without new model outputs."""
    manifest = R.check(out, G.STAGES + ("ordinal_seal.json",))
    obs = R.prepared(out)
    d = H.selected(out / "analysis.npz", ("odd",))["odd"]
    rows = R.read(out / "ordinal_feasibility.json")["checks"]
    evidence, consistent = [], True
    for row in rows:
        q = manifest["config"]["targets"].index(row["target"])
        adj = np.zeros((24, 24), int)
        magnitudes = np.zeros((24, 24))
        for k, (i, j) in enumerate(obs["edges"]):
            value = d[k, q]
            if abs(value) > row["margin"]:
                u, v = (i, j) if value > 0 else (j, i)
                adj[u, v] = 1
                magnitudes[u, v] = abs(value)
        order, cycle = topological_or_cycle(adj)
        feasible = bool(order)
        consistent &= feasible == row["strict_scalar_order_feasible_on_observed_edges"]
        if cycle:
            weights = [float(magnitudes[i, j]) for i, j in pairwise(cycle)]
            if not all(w > row["margin"] for w in weights):
                raise ValueError("Cycle witness uses an unobserved/non-strict edge")
        else:
            weights = []
            position = np.argsort(order)
            if any(
                position[i] >= position[j]
                for i, j in zip(*np.nonzero(adj), strict=True)
            ):
                raise ValueError("Topological witness violates a comparison")
        evidence.append(
            {
                "target": row["target"],
                "margin": row["margin"],
                "feasible": feasible,
                "topological_witness": order,
                "directed_cycle_witness": cycle,
                "cycle_absolute_edges": weights,
                "cycle_min_absolute_edge": min(weights) if weights else None,
            }
        )
    b = H.selected(out / "analysis.npz", ("incidence",))["incidence"]
    smallest = float(np.linalg.svd(b, compute_uv=False)[-2])
    tolerance = manifest["config"]["observable_contract"]["output_tolerance"]
    result = {
        "all10_SCC_Kahn_agree": bool(consistent),
        "witnesses": evidence,
        "incidence_smallest_nonzero_singular": smallest,
        "conditional_position_L2_perturbation_bound": float(
            np.sqrt(32) * tolerance / smallest
        ),
        "bound_condition": "IF every observed edge perturbation<=1e-4;two compatibility checks do NOT prove this uniformly",
        "new_model_API_train_private_calls": 0,
        "same_agent_not_blind": True,
        "PASS": bool(consistent),
    }
    R.write(out / "ordinal_verification.json", result)
    R.write(
        out / "ordinal_verifier_provenance.json",
        {"script": R.digest(Path(__file__)), "no_primary_science_change": True},
    )
    R.seal(
        out,
        "ordinal_verification_seal.json",
        ("ordinal_verification.json", "ordinal_verifier_provenance.json"),
    )
    if not consistent:
        raise ValueError("Ordinal verification failed;preserve outcomes")
    print("HM228 independent ordinal witness verification PASS", flush=True)


def main():
    """Verify once; a failed verification is not automatically retried."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=C.DEFAULT)
    args = parser.parse_args()
    verify(args.out)


if __name__ == "__main__":
    main()
