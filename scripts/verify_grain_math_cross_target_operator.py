#!/usr/bin/env python3
"""HM60: raw-cell replay, graph least squares and independent spectral checks."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_construct_routing import write
from scripts.run_grain_math_cross_target_operator import OUT, check
from scripts.run_grain_math_event_readout_composition import check as source_check
from scripts.run_grain_math_event_readout_composition import observation


def table(out, name):
    """Read a complete saved table without recomputing it with production formulas."""
    with (out / "03" / (name + ".csv")).open(newline="") as stream:
        return list(csv.DictReader(stream))


def independently_construct(rows, targets, regimes, unit):
    """Rebuild contrasts from raw signed cells, not production matrix constructors."""
    ix = {
        (
            r["role"],
            r["regime"],
            r["family"],
            r["polarity"],
            r["order"],
            r["content"],
            r["query"],
        ): r[unit]
        for r in rows
    }
    result = {}
    for family in range(2):
        result[f"atom/f{family}"] = np.array(
            [
                [
                    sum(
                        (1 if p else -1)
                        * ix["atom", "single_event", family, p, -1, c, q]
                        for p in range(2)
                    )
                    for c in targets
                ]
                for q in targets
            ]
        )
    result["atom/pooled"] = np.stack([result["atom/f0"], result["atom/f1"]]).mean(
        axis=0
    )
    for regime in regimes:
        for family, order in itertools.product(range(2), repeat=2):
            result[f"account/{regime}/f{family}/o{order}"] = np.array(
                [
                    [
                        sum(
                            (1 if p else -1)
                            * ix["account", regime, family, p, order, c, q]
                            for p in range(2)
                        )
                        for c in targets
                    ]
                    for q in targets
                ]
            )
        result[f"account/{regime}/pooled"] = np.stack(
            [
                result[f"account/{regime}/f{f}/o{o}"]
                for f, o in itertools.product(range(2), repeat=2)
            ]
        ).mean(axis=0)
    for name, high, low in (
        ("alias_minus_balanced", "alias_focus", "balanced"),
        ("distinct_minus_alias", "distinct_focus", "alias_focus"),
        ("other_minus_balanced", "other_focus", "balanced"),
    ):
        for f, o in itertools.product(range(2), repeat=2):
            result[f"update/{name}/f{f}/o{o}"] = (
                result[f"account/{high}/f{f}/o{o}"] - result[f"account/{low}/f{f}/o{o}"]
            )
        result[f"update/{name}/pooled"] = np.stack(
            [
                result[f"update/{name}/f{f}/o{o}"]
                for f, o in itertools.product(range(2), repeat=2)
            ]
        ).mean(axis=0)
    for regime in regimes:
        for f in range(2):
            result[f"order/{regime}/f{f}"] = (
                result[f"account/{regime}/f{f}/o1"]
                - result[f"account/{regime}/f{f}/o0"]
            )
        result[f"order/{regime}/pooled"] = np.stack(
            [result[f"order/{regime}/f{f}"] for f in range(2)]
        ).mean(axis=0)
    return result


def verify(out):
    """Verify real readouts and every saved numerical table using alternate formulas."""
    m = check(out)
    if read_json(out / "run_status.json")["status"] != "ANALYZED":
        raise ValueError("Complete analysis required")
    for name, sha in read_json(out / "03/artifact_hashes.json").items():
        if digest(out / "03" / name) != sha:
            raise ValueError("Scientific output drift " + name)
    source = ROOT / m["spec"]["source"]
    source_manifest = source_check(source)
    original = {r["unit"]: r for r in read_json(source / "00/rows.json")}
    rows = read_json(out / "00/observations.json")
    if len(rows) != 1296 or {r["unit"] for r in rows} != set(original):
        raise ValueError("Complete unique raw snapshot required")
    maximum = 0.0

    def equal(a, b):
        nonlocal maximum
        if a is None:
            if b not in (None, ""):
                raise ValueError("Undefined statistic replaced with a number")
            return
        error = float(
            np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))
        )
        if not np.isfinite(error):
            raise ValueError("Nonfinite verification difference")
        maximum = max(maximum, error)

    for row in rows:
        o = original[row["unit"]]
        if any(
            row[k] != o[k]
            for k in (
                "unit",
                "parent_unit",
                "role",
                "content",
                "query",
                "family",
                "polarity",
                "order",
                "regime",
            )
        ):
            raise ValueError("Raw identity changed")
        raw, _ = observation(source, o, source_manifest)
        if o["reused"]:
            source_path = (
                ROOT
                / source_manifest["spec"]["parent"]
                / source_manifest["reused"][str(o["parent_unit"])]["meta"]
            )
        else:
            source_path = source / f"01/units/unit_{o['unit']:04d}.json"
        if row["source_meta"] != str(source_path.relative_to(ROOT)):
            raise ValueError("Raw artifact address mismatch")
        z = np.asarray(raw["candidate_logits"], float)
        equal(z, row["candidate_logits"])
        score = float(softmax(z) @ np.linspace(0, 1, 10))
        log_odds = float(logsumexp(z[5:]) - logsumexp(z[:5]))
        equal(score, row["score"])
        equal(log_odds, row["log_odds"])
        equal(raw["candidate_mass"], row["candidate_mass"])
        row.update(score=score, log_odds=log_odds)
    spec = m["spec"]
    targets, regimes = spec["targets"], spec["regimes"]
    operators = {
        (unit, name): d
        for unit in spec["units"]
        for name, d in independently_construct(rows, targets, regimes, unit).items()
    }
    entries = table(out, "entries")
    seen = set()
    saved = {}
    for r in entries:
        key = r["unit"], r["operator"], r["query"], r["content"]
        if key in seen:
            raise ValueError("Duplicate saved matrix entry")
        seen.add(key)
        d = operators[key[:2]]
        equal(d[targets.index(key[2]), targets.index(key[3])], r["response"])
        saved[key] = r
    expected = {
        (u, n, q, c)
        for u, n in operators
        for q, c in itertools.product(targets, repeat=2)
    }
    if seen != expected:
        raise ValueError("Incomplete saved operator roster")
    b = np.zeros((36, 12))
    for edge in range(36):
        b[edge, edge // 6], b[edge, 6 + edge % 6] = 1, -1
    gradients, cycles = {}, {}
    for key, d in operators.items():
        g = (b @ np.linalg.lstsq(b, d.ravel(), rcond=None)[0]).reshape(6, 6)
        c = d - g
        gradients[key], cycles[key] = g, c
        for q, content in itertools.product(range(6), repeat=2):
            r = saved[*key, targets[q], targets[content]]
            equal(g[q, content], r["gradient"])
            equal(c[q, content], r["cycle"])
        equal(b.T @ c.ravel(), np.zeros(12))
    spectrum_rows = table(out, "spectra")
    if {(r["unit"], r["operator"], int(r["index"])) for r in spectrum_rows} != {
        (*key, index) for key in operators for index in range(1, 7)
    }:
        raise ValueError("Spectrum identity coverage mismatch")
    for key, d in operators.items():
        reported = sorted(
            [r for r in spectrum_rows if (r["unit"], r["operator"]) == key],
            key=lambda r: int(r["index"]),
        )
        eigen = np.linalg.eigvalsh(d @ d.T)[::-1]
        equal(np.maximum(eigen, 0), [r["energy"] for r in reported])
        equal(
            [float(r["singular_value"]) ** 2 for r in reported],
            [r["energy"] for r in reported],
        )
    rectangle_rows = table(out, "rectangles")
    if {
        (r["unit"], r["operator"], *(int(r[k]) for k in ("q0", "q1", "c0", "c1")))
        for r in rectangle_rows
    } != {
        (*key, *qs, *cs)
        for key in operators
        for qs, cs in itertools.product(itertools.combinations(range(6), 2), repeat=2)
    }:
        raise ValueError("Rectangle identity coverage mismatch")
    for r in rectangle_rows:
        q0, q1, c0, c1 = (int(r[k]) for k in ("q0", "q1", "c0", "c1"))
        vector = np.zeros(36)
        (
            vector[6 * q0 + c0],
            vector[6 * q0 + c1],
            vector[6 * q1 + c0],
            vector[6 * q1 + c1],
        ) = 1, -1, -1, 1
        equal(vector @ operators[r["unit"], r["operator"]].ravel(), r["contrast"])
        equal(b.T @ vector, np.zeros(12))
    assignment_rows = table(out, "assignments")
    perm_seen = set()
    assignment_values = {}
    for r in assignment_rows:
        key = r["unit"], r["operator"]
        p = tuple(map(int, r["permutation"]))
        if sorted(p) != list(range(6)) or (*key, p) in perm_seen:
            raise ValueError("Invalid/duplicate matching")
        perm_seen.add((*key, p))
        v = float(np.sum(operators[key] * np.eye(6)[list(p)]))
        equal(v, r["trace"])
        assignment_values.setdefault(key, []).append(v)
    summary_rows = table(out, "operator_summary")
    if {(r["unit"], r["operator"]) for r in summary_rows} != set(operators):
        raise ValueError("Summary identity coverage mismatch")
    for r in summary_rows:
        key = r["unit"], r["operator"]
        d, g, c = operators[key], gradients[key], cycles[key]
        energy = float(d.ravel() @ d.ravel())
        eigen = np.maximum(np.linalg.eigvalsh(d.T @ d)[::-1], 0)
        values = np.asarray(assignment_values[key])
        trace = float(np.trace(d))
        off_mean = (d.sum() - trace) / 30
        a = operators[r["unit"], "atom/pooled"]
        ae = float(a.ravel() @ a.ravel())
        active = np.sqrt(energy) > spec["numerical_zero_norm"]
        ref_active = np.sqrt(ae) > spec["numerical_zero_norm"]
        alpha = float(d.ravel() @ a.ravel() / ae) if ref_active else None
        rem = d - alpha * a if ref_active else None
        expected_values = {
            "energy": energy,
            "rank1_energy_share": float(eigen[0] / energy) if active else None,
            "rank1_tail_share": float(eigen[1:].sum() / energy) if active else None,
            "cycle_energy_share": float(c.ravel() @ c.ravel() / energy)
            if active
            else None,
            "gradient_energy_share": float(g.ravel() @ g.ravel() / energy)
            if active
            else None,
            "named_trace": trace,
            "strictly_better_assignments": int(
                np.sum(values > trace + spec["assignment_tie_tolerance"])
            ),
            "tied_assignments": int(
                np.sum(np.abs(values - trace) <= spec["assignment_tie_tolerance"])
            ),
            "named_rank": 1
            + int(np.sum(values > trace + spec["assignment_tie_tolerance"])),
            "diagonal_mean": trace / 6,
            "off_diagonal_mean": float(off_mean),
            "named_diagonal_excess": trace / 6 - float(off_mean),
            "positive_diagonals": int(np.sum(np.diag(d) > 0)),
            "diagonal_row_winners": int(np.sum(np.diag(d) >= d.max(axis=1) - 1e-12)),
            "atomic_scale": alpha,
            "shape_remainder_rms": float(np.sqrt(np.mean(rem**2)))
            if ref_active
            else None,
            "shape_remainder_share": float(np.sum(rem**2) / energy)
            if ref_active and active
            else None,
            "total_assignments": 720,
            "assignment_maximum": float(values.max()),
            "assignment_minimum": float(values.min()),
        }
        for field, value in expected_values.items():
            equal(value, r[field])
    family_rows = table(out, "family_comparisons")
    family_names = (
        {"atom"}
        | {"account/" + r for r in regimes}
        | {
            "update/alias_minus_balanced",
            "update/distinct_minus_alias",
            "update/other_minus_balanced",
        }
    )
    if {(r["unit"], r["comparison"]) for r in family_rows} != {
        (unit, name) for unit in spec["units"] for name in family_names
    }:
        raise ValueError("Family comparison identity coverage mismatch")
    for r in family_rows:
        unit, name = r["unit"], r["comparison"]
        if name == "atom":
            a, d = (operators[unit, f"atom/f{f}"] for f in range(2))
        else:
            a, d = (
                np.stack([operators[unit, f"{name}/f{f}/o{o}"] for o in range(2)]).mean(
                    axis=0
                )
                for f in range(2)
            )

        def cos(x, y):
            nx, ny = np.sqrt(np.sum(x * x)), np.sqrt(np.sum(y * y))
            return (
                None
                if min(nx, ny) <= spec["numerical_zero_norm"]
                else float(np.sum(x * y) / (nx * ny))
            )

        ac = a - (b @ np.linalg.lstsq(b, a.ravel(), rcond=None)[0]).reshape(6, 6)
        dc = d - (b @ np.linalg.lstsq(b, d.ravel(), rcond=None)[0]).reshape(6, 6)
        equal(cos(a, d), r["operator_cosine"])
        equal(cos(ac, dc), r["cycle_cosine"])
        equal(float(np.sqrt(np.mean((a - d) ** 2))), r["difference_rms"])
        equal(int(np.sum(np.diag(a) * np.diag(d) > 0)), r["diagonal_sign_matches"])
    counts = {
        n: len(table(out, n))
        for n in (
            "entries",
            "spectra",
            "rectangles",
            "assignments",
            "operator_summary",
            "family_comparisons",
        )
    }
    if counts != read_json(out / "03/summary.json")["rows"] or counts != {
        "entries": 3600,
        "spectra": 600,
        "rectangles": 22500,
        "assignments": 72000,
        "operator_summary": 100,
        "family_comparisons": 16,
    }:
        raise ValueError("Numerical output coverage mismatch")
    if maximum > spec["arithmetic_tolerance"]:
        raise ValueError("Alternate formula mismatch " + str(maximum))
    result = {
        "status": "VERIFIED_FINITE_HM60_NOT_GLOBAL",
        "raw_observations": len(rows),
        "table_rows": counts,
        "maximum_alternate_error": maximum,
        "new_forwards": 0,
        "fit_or_select_predictor": False,
        "incidence_rank": int(np.linalg.matrix_rank(b)),
        "cycle_space_dimension": 36 - int(np.linalg.matrix_rank(b)),
        "independent_method_replication": False,
        "original_scope_complete": False,
    }
    write(out / "verification.json", result)
    print(result, flush=True)


def main():
    """Audit completed HM60 without changing any source scientific artifact."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    verify(parser.parse_args().output)


if __name__ == "__main__":
    main()
