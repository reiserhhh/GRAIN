#!/usr/bin/env python3
"""HM34 alternate primal SVD and complete paired-response/action replay."""

from __future__ import annotations

import argparse
import itertools
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction.math_crossed_field import directions
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts import run_grain_math_query_context as OLD
from scripts.run_grain_math_position_support import array_unit, check, now
from scripts.run_grain_math_response import write_json
from scripts.verify_grain_math_crossed_field import table

PARENT = ROOT / "results/grain_math_response_v1/early_query_response_operator_r1"
OUT = ROOT / "results/grain_math_response_v1/early_query_response_verification_r1"


def primal(d, g):
    """Use thin SVD of physical D, not the primary Gram-eigendecomposition."""
    u, s, vt = np.linalg.svd(d, full_matrices=False)
    threshold = 100 * np.finfo(float).eps * d.shape[1] * s[0] ** 2
    keep = s * s > threshold
    return u[:, keep], s[keep], vt[keep].T, g


def stats(y, p):
    """Alternate norm-based metric calculation including the zero reference."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    n = y.size
    rms, err = np.linalg.norm(y) / np.sqrt(n), np.linalg.norm(p - y) / np.sqrt(n)
    return {
        "values": n,
        "actual_rms": rms,
        "prediction_rms": np.linalg.norm(p) / np.sqrt(n),
        "rmse": err,
        "relative_rmse": err / rms if rms else None,
        "zero_response_rmse": rms,
        "sign_agreement": sum((y * p > 0).ravel()),
        "actual_nonzero": sum((y != 0).ravel()),
    }


def verify(parent, out):
    """Rebuild all source fits and held comparisons without a model invocation."""
    if out.exists():
        raise FileExistsError("preserve earlier verification")
    m = check(parent)
    spec = m["spec"]
    ancestor = ROOT / spec["parent"]
    am = read_json(ancestor / "00/manifest.json")
    ledger = read_json(parent / "02/artifact_hashes.json")
    for name, sha in ledger.items():
        if digest(parent / name) != sha:
            raise ValueError("frozen output changed " + name)
    status = read_json(parent / "run_status.json")
    if status != {"status": "EVALUATED", "source_fits": 3, "new_model_calls": 0}:
        raise ValueError("fixed stage/call counts required")
    barrier_path = parent / "01/source_barrier.json"
    barrier = read_json(barrier_path)
    summary = read_json(parent / "02/summary.json")
    freeze = datetime.fromisoformat(barrier["frozen_at"])
    if (
        not datetime.fromisoformat(m["frozen_before_outputs"])
        < freeze
        < datetime.fromisoformat(summary["evaluated_at"])
    ):
        raise ValueError("source freeze before held scoring required")
    for name, sha in barrier["files"].items():
        if digest(parent / name) != sha:
            raise ValueError("source model changed")
    design = read_json(parent / "00/design.json")
    if len(design) != 768 or any(
        r["pair"] not in spec["roles"][r["role"]] for r in design
    ):
        raise ValueError("original complete roles required")
    coef = {
        (int(r["pair"]), int(r["wording"]), r["query"]): r
        for r in table(parent / "02/coefficients.csv")
    }
    fixed_table = {
        (
            int(r["pair"]),
            int(r["wording"]),
            r["query"],
            int(r["H"]),
            int(r["B"]),
            int(r["Q"]),
        ): r
        for r in table(parent / "02/fixed_responses.csv")
    }
    action_table = {
        (*key, r["arm"]): r
        for r in table(parent / "02/actions.csv")
        for key in [
            (
                int(r["pair"]),
                int(r["wording"]),
                r["query"],
                int(r["H"]),
                int(r["B"]),
                int(r["Q"]),
            )
        ]
    }
    metrics = {
        (r["role"], r["query"], r["arm"]): r for r in table(parent / "02/metrics.csv")
    }
    old_actions = {
        (int(r["unit"]), r["arm"]): float(r["odd_action"])
        for r in table(ancestor / "02/actions.csv")
    }
    errors = {
        k: 0.0
        for k in (
            "source",
            "coefficient",
            "array",
            "action",
            "metric",
            "character",
            "identity",
        )
    }

    def compare(kind, a, b):
        if a is None:
            if b not in (None, ""):
                raise ValueError("undefined quantity fabricated")
            return
        difference = float(np.max(abs(np.asarray(a, float) - np.asarray(b, float))))
        if not np.isfinite(difference):
            raise ValueError("nonfinite alternate replay")
        errors[kind] = max(errors[kind], difference)

    def read_group(pair, w, q):
        rows = [
            r for r in design if (r["pair"], r["wording"], r["query"]) == (pair, w, q)
        ]
        by = {
            (
                r["donor_side"],
                r["receiver_side"],
                r["early_query_side"],
                r["query_side"],
            ): r
            for r in rows
        }
        if len(by) != 16 or len(rows) != 16:
            raise ValueError("sixteen distinct corners required")
        first = by[0, 0, 0, 0]
        ids = first["early_query_token_ids"]
        if any(r["early_query_token_ids"] != ids for r in rows):
            raise ValueError("ordered query addresses changed")
        pp = []
        for pside in (0, 1):
            r = by[0, 0, pside, 0]
            cap = (
                ancestor
                / "01/captures"
                / f"unit_{r['early_query_donor_unit']:03d}.json"
            )
            meta = array_unit(cap)
            if (
                meta["early_query_token_ids"] != ids
                or meta["address"]["role"] != first["role"]
            ):
                raise ValueError("natural carrier ownership changed")
            with np.load(cap.with_suffix(".npz"), allow_pickle=False) as bank:
                pp.append(bank["p"].astype(float))
        vectors, h = {}, {}
        for bits, r in by.items():
            path, _ = OLD.paths(ancestor, r, am)
            array_unit(path)
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as bank:
                vectors[bits] = bank["gradient"].astype(float)
                h[bits[0]] = bank["h"].copy()
        shape = vectors[0, 0, 0, 0].shape
        character = sum(((-1) ** bits[2]) * v for bits, v in vectors.items()) / 16
        fixed = {
            (i, j, k): (vectors[i, j, 0, k] - vectors[i, j, 1, k]) / 2
            for i, j, k in itertools.product(range(2), repeat=3)
        }
        return {
            "d": ((pp[0] - pp[1]) / 2).ravel(),
            "g": character.ravel(),
            "fixed": fixed,
            "h": h,
            "by": by,
            "shape": shape,
            "ids": ids,
            "p_shape": pp[0].shape,
            "role": first["role"],
        }

    checked = []
    with (
        np.load(parent / "02/response_arrays.npz", allow_pickle=False) as saved,
        np.load(ancestor / "00/directions.npz", allow_pickle=False) as bank,
    ):
        for q in spec["queries"]:
            source = [read_group(pair, w, q) for pair in (0, 1) for w in range(4)]
            d, g = (
                np.column_stack([r["d"] for r in source]),
                np.column_stack([r["g"] for r in source]),
            )
            u, s, v, target = primal(d, g)
            path = parent / "01/models" / f"{q}.json"
            model = read_json(path)
            if (
                model["rank"] != len(s)
                or model["held_numeric_readouts_used"]
                or model["source_keys"] != [[p, w] for p in (0, 1) for w in range(4)]
            ):
                raise ValueError("source-only law or fixed numerical rank changed")
            if (
                not datetime.fromisoformat(model["fitted_at"]) < freeze
                or digest(path.with_suffix(".npz")) != model["array_sha256"]
            ):
                raise ValueError("source chronology/hash changed")
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as original:
                compare("source", d, original["D"])
                compare("source", g, original["G"])
                compare(
                    "source",
                    s[0] ** 2 * 100 * np.finfo(float).eps * d.shape[1],
                    model["tolerance"],
                )
                compare("source", np.linalg.norm(g), model["source_target_norm"])
                compare(
                    "source",
                    np.linalg.norm(target @ v @ v.T - g),
                    model["source_interpolation_error"],
                )
                compare(
                    "source", target @ v @ v.T, g @ original["gram_inverse"] @ (d.T @ d)
                )
            arms = directions(bank[q + "_b"])
            for pair in range(4):
                for w in range(4):
                    record = read_group(pair, w, q)
                    if (
                        record["ids"] != model["query_token_ids"]
                        or list(record["p_shape"]) != model["p_shape"]
                    ):
                        raise ValueError("query spaces silently combined")
                    x = record["d"]
                    weights = v @ ((u.T @ x) / s)
                    predicted = (g @ weights).reshape(record["shape"])
                    actual = record["g"].reshape(record["shape"])
                    unsupported = x - u @ (u.T @ x)
                    key = f"p{pair}_w{w}_{q}"
                    reported = coef[pair, w, q]
                    if (
                        int(reported["source_rank"]) != len(s)
                        or reported["role"] != record["role"]
                    ):
                        raise ValueError("reported source rank/role changed")
                    with np.load(
                        ancestor / "02/cube_arrays.npz", allow_pickle=False
                    ) as characters:
                        mismatch = float(
                            np.max(
                                abs(
                                    actual
                                    - characters[key + "_gradient_none_coefficients"][2]
                                )
                            )
                        )
                        compare(
                            "character", mismatch, reported["character_identity_error"]
                        )
                    for k, value in (
                        ("input", x),
                        ("common", actual),
                        ("predicted", predicted),
                        ("weights", weights),
                        ("unsupported", unsupported),
                    ):
                        compare("array", value, saved[key + "_" + k])
                    for name, value in stats(actual, predicted).items():
                        compare("coefficient", value, reported[name])
                    for name, value in (
                        ("input_norm", np.linalg.norm(x)),
                        ("supported_norm", np.linalg.norm(u.T @ x)),
                        ("unsupported_norm", np.linalg.norm(unsupported)),
                        (
                            "unsupported_ratio",
                            np.linalg.norm(unsupported) / np.linalg.norm(x),
                        ),
                    ):
                        compare("coefficient", value, reported[name])
                    for arm in spec["arms"]:
                        direction = arms[arm]
                        for suffix, val in (
                            ("_actual", actual),
                            ("_predicted", predicted),
                        ):
                            compare(
                                "coefficient",
                                np.sum(val * direction / np.linalg.norm(direction)),
                                reported[arm + suffix],
                            )
                    for (hside, bside, qside), fixed in record["fixed"].items():
                        address = (pair, w, q, hside, bside, qside)
                        for name, value in stats(fixed, predicted).items():
                            compare("coefficient", value, fixed_table[address][name])
                        if fixed_table[address]["role"] != record["role"]:
                            raise ValueError("fixed-response role changed")
                        compare(
                            "array", fixed, saved[key + "_fixed"][hside, bside, qside]
                        )
                        for arm in spec["arms"]:
                            units = [
                                record["by"][hside, bside, p, qside]["unit"]
                                for p in (0, 1)
                            ]
                            y = (
                                old_actions[units[0], arm] - old_actions[units[1], arm]
                            ) / 2
                            h = record["h"][hside].astype(np.float32)
                            step = (
                                arms[arm]
                                * (
                                    am["spec"]["radius_relative_span_norm"]
                                    * np.linalg.norm(h)
                                    / np.linalg.norm(arms[arm])
                                )
                            ).astype(np.float32)
                            delta = (
                                (h + step).astype(float) - (h - step).astype(float)
                            ) / 2
                            p, c, f = [
                                float(np.sum(val * delta))
                                for val in (predicted, actual, fixed)
                            ]
                            output = {
                                "actual": y,
                                "predicted": p,
                                "common_local": c,
                                "fixed_local": f,
                                "operator_error": p - c,
                                "interaction_error": c - f,
                                "finite_error": f - y,
                                "physical_delta_norm": np.linalg.norm(delta),
                                "identity_error": abs(
                                    p - y - ((p - c) + (c - f) + (f - y))
                                ),
                            }
                            reported_action = action_table[(*address, arm)]
                            if reported_action["role"] != record["role"]:
                                raise ValueError("action role changed")
                            if [
                                int(reported_action[n])
                                for n in ("left_unit", "right_unit")
                            ] != units:
                                raise ValueError("native comparison addresses changed")
                            for name, value in output.items():
                                compare("action", value, reported_action[name])
                            checked.append(
                                {
                                    "role": record["role"],
                                    "query": q,
                                    "arm": arm,
                                    **output,
                                }
                            )
    for (role, q, arm), row in metrics.items():
        data = [
            r for r in checked if (r["role"], r["query"], r["arm"]) == (role, q, arm)
        ]
        y = np.array([r["actual"] for r in data])
        for method in ("predicted", "common_local", "fixed_local"):
            for name, value in stats(y, [r[method] for r in data]).items():
                compare("metric", value, row[method + "_" + name])
        selected = [r for key, r in coef.items() if r["role"] == role and key[2] == q]
        compare("metric", len(selected), row["cubes"])
        compare(
            "metric",
            np.mean([float(r["unsupported_ratio"]) for r in selected]),
            row["unsupported_ratio_mean"],
        )
        for name, value in stats(
            [float(r[arm + "_actual"]) for r in selected],
            [float(r[arm + "_predicted"]) for r in selected],
        ).items():
            compare("metric", value, row["coefficient_" + name])
        parts = np.array(
            [
                [r[k] for k in ("operator_error", "interaction_error", "finite_error")]
                for r in data
            ]
        )
        total = 0.0
        for i, name in enumerate(("operator", "interaction", "finite")):
            value = np.dot(parts[:, i], parts[:, i]) / len(data)
            compare("metric", value, row[name + "_mse"])
            total += value
            for j in range(i + 1, 3):
                value = 2 * np.dot(parts[:, i], parts[:, j]) / len(data)
                compare("metric", value, row[f"cross_{i}_{j}"])
                total += value
        compare(
            "identity",
            abs(np.mean((np.array([r["predicted"] for r in data]) - y) ** 2) - total),
            row["error_mse_identity"],
        )
    counts = {
        "source_fits": 3,
        "coefficient_rows": len(coef),
        "fixed_gradient_rows": len(fixed_table),
        "finite_action_contrasts": len(action_table),
        "metric_rows": len(metrics),
        "new_model_calls": 0,
    }
    if (
        counts != summary["counts"]
        or list(counts.values()) != [3, 48, 384, 768, 18, 0]
        or max(errors.values()) > 1e-8
    ):
        raise ValueError("incomplete or inaccurate alternate replay " + str(errors))
    out.mkdir(parents=True)
    write_json(
        out / "summary.json",
        {
            "status": "VERIFIED_FINITE_HM34_NOT_GLOBAL",
            "counts": counts,
            "maximum_errors": errors,
            "absolute_tolerance": 1e-8,
            "artifact_hashes_checked": len(ledger),
            "inherited_hashes_checked": len(m["files"]),
            "source_barrier_sha256": digest(barrier_path),
            "checked_at": now(),
            "independence": "same-agent primal SVD, original gradient character and addressed finite-action replay; not independent human validation",
            "new_model_calls": 0,
            "new_fits": 0,
        },
    )
    write_json(
        out / "artifact_hashes.json", {"summary.json": digest(out / "summary.json")}
    )
    print(read_json(out / "summary.json"), flush=True)


def main():
    """Verify a frozen packet into a fresh directory only."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    verify(args.parent, args.out)


if __name__ == "__main__":
    main()
