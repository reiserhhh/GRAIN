#!/usr/bin/env python3
"""Result-after HM60 diagnostic: where does named-target alignment reside?"""

from __future__ import annotations

import itertools
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_cross_target_operator as M
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_construct_routing import write
from scripts.run_grain_math_cross_target_operator import OUT, check
from scripts.run_grain_math_response import write_csv
from scripts.verify_grain_math_cross_target_operator import table

CODE = (
    "docs/GRAIN_CROSS_TARGET_RANK1_ALIGNMENT_FOLLOWUP_V1.md",
    "scripts/analyze_grain_math_cross_target_rank1_alignment.py",
    "tests/test_grain_math_cross_target_rank1_alignment.py",
)


def leading_component(value):
    """Return the fixed leading component only when its subspace is uniquely defined."""
    d = M.finite_matrix(value)
    u, s, v = np.linalg.svd(d, full_matrices=False)
    if s[0] <= M.ZERO_NORM or s[0] - s[1] <= M.ZERO_NORM:
        return None, None, float(s[0] - s[1])
    first = np.outer(u[:, 0], v[0]) * s[0]
    return first, d - first, float(s[0] - s[1])


def diagnostics(value):
    """Describe label alignment without treating its rank as a sampling p-value."""
    d = M.finite_matrix(value)
    diagonal = float(np.trace(d) / 6)
    off = float((d.sum() - np.trace(d)) / 30)
    cycle = M.graph_split(d)[1]
    return M.assignment(d)[2] | {
        "energy": float(np.sum(d * d)),
        "cycle_energy": float(np.sum(cycle * cycle)),
        "diagonal_mean": diagonal,
        "off_diagonal_mean": off,
        "named_diagonal_excess": diagonal - off,
    }


def run(out=OUT):
    """Freeze a separate companion and audit its decomposition without changing HM60."""
    check(out)
    if (
        read_json(out / "verification.json")["status"]
        != "VERIFIED_FINITE_HM60_NOT_GLOBAL"
    ):
        raise ValueError("Verified primary HM60 required")
    for name, sha in read_json(out / "03/artifact_hashes.json").items():
        if digest(out / "03" / name) != sha:
            raise ValueError("Primary scientific output drift")
    dest = out / "04_rank1_alignment"
    if dest.exists():
        raise FileExistsError("Preserve result-after companion")
    (dest / "code").mkdir(parents=True)
    hashes = {}
    for name in CODE:
        path = dest / "code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
        hashes[name] = digest(path)
    inputs = {
        n: digest(out / n)
        for n in ("03/entries.csv", "03/operator_summary.csv", "verification.json")
    }
    write(
        dest / "manifest.json",
        {
            "code": hashes,
            "source": inputs,
            "result_after": True,
            "operators": 100,
            "new_forwards": 0,
        },
    )
    operators = {}
    for row in table(out, "entries"):
        key = row["unit"], row["operator"]
        d = operators.setdefault(key, np.zeros((6, 6)))
        d[M.TARGETS.index(row["query"]), M.TARGETS.index(row["content"])] = float(
            row["response"]
        )
    stats, entries, audit = [], [], []
    maximum = 0.0
    permutations = np.asarray(list(itertools.permutations(range(6))))
    b = M.incidence()
    for (unit, name), d in operators.items():
        first, remainder, gap = leading_component(d)
        tags = {"unit": unit, "operator": name}
        components = {"full": d}
        if first is not None:
            components.update(leading=first, remainder=remainder)
            eigenvalues, v = np.linalg.eigh(d.T @ d)
            alternative = d @ np.outer(v[:, -1], v[:, -1])
            error = float(np.max(np.abs(first - alternative)))
            reconstruction = float(np.max(np.abs(d - first - remainder)))
            orthogonality = float(abs(np.sum(first * remainder)))
            maximum = max(maximum, error, reconstruction, orthogonality)
            audit.append(
                tags
                | {
                    "leading_gap": gap,
                    "gram_gap": float(eigenvalues[-1] - eigenvalues[-2]),
                    "alternate_error": error,
                    "reconstruction_error": reconstruction,
                    "orthogonality_error": orthogonality,
                }
            )
        else:
            audit.append(
                tags
                | {
                    "leading_gap": gap,
                    "gram_gap": None,
                    "alternate_error": None,
                    "reconstruction_error": None,
                    "orthogonality_error": None,
                }
            )
        for component, x in components.items():
            label = tags | {
                "component": component,
                "leading_alignment_available": first is not None,
            }
            result = diagnostics(x)
            # Independent named trace and finite label enumeration, not another fitted model.
            actual_trace = sum(float(x[i, i]) for i in range(6))
            maximum = max(maximum, abs(result["named_trace"] - actual_trace))
            alternative_traces = x[np.arange(6)[None, :], permutations].sum(axis=1)
            better = int(np.sum(alternative_traces > actual_trace + 1e-12))
            ties = int(np.sum(np.abs(alternative_traces - actual_trace) <= 1e-12))
            if (
                better != result["strictly_better_assignments"]
                or ties != result["tied_assignments"]
            ):
                raise ValueError("Alternate label rank disagrees")
            graph_residual = (
                x.ravel() - b @ np.linalg.lstsq(b, x.ravel(), rcond=None)[0]
            )
            maximum = max(
                maximum,
                abs(result["cycle_energy"] - float(graph_residual @ graph_residual)),
            )
            stats.append(label | result)
            for q in range(6):
                for c in range(6):
                    entries.append(
                        label
                        | {
                            "query": M.TARGETS[q],
                            "content": M.TARGETS[c],
                            "response": float(x[q, c]),
                        }
                    )
        if first is not None:
            excess = [
                diagnostics(x)["named_diagonal_excess"] for x in (d, first, remainder)
            ]
            maximum = max(maximum, abs(excess[0] - excess[1] - excess[2]))
    if len(operators) != 100 or maximum > 1e-10:
        raise ValueError("Companion coverage/alternate arithmetic failure")
    write_csv(dest / "components.csv", entries)
    write_csv(dest / "alignment.csv", stats)
    write_csv(dest / "alternate_checks.csv", audit)
    for name, sha in hashes.items():
        if digest(ROOT / name) != sha or digest(dest / "code" / name) != sha:
            raise ValueError("Frozen companion code drift")
    for name, sha in inputs.items():
        if digest(out / name) != sha:
            raise ValueError("Frozen primary input drift")
    summary = {
        "status": "VERIFIED_RESULT_AFTER_HM60_RANK1_ALIGNMENT",
        "operators": len(operators),
        "component_rows": len(stats),
        "entry_rows": len(entries),
        "maximum_alternate_error": maximum,
        "unique_leading_operators": sum(
            r["alternate_error"] is not None for r in audit
        ),
        "new_forwards": 0,
        "new_fit": False,
        "original_scope_complete": False,
        "atom_components": [r for r in stats if r["operator"].startswith("atom/")],
        "pooled_components": [r for r in stats if r["operator"].endswith("/pooled")],
    }
    write(dest / "summary.json", summary)
    write(
        dest / "artifact_hashes.json",
        {
            p.name: digest(p)
            for p in sorted(dest.iterdir())
            if p.is_file() and not p.name.startswith("._")
        },
    )
    print(
        {
            k: summary[k]
            for k in (
                "status",
                "operators",
                "component_rows",
                "entry_rows",
                "maximum_alternate_error",
            )
        },
        flush=True,
    )


if __name__ == "__main__":
    run()
