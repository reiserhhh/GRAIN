#!/usr/bin/env python3
"""HM32 alternate product projections, score replay and conditional-variance bounds."""

from __future__ import annotations

import argparse
import itertools
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_position_support import check
from scripts.run_grain_math_response import write_json
from scripts.verify_grain_math_crossed_field import table
from scripts.verify_grain_math_position_support import rational_inverse, score

PARENT = ROOT / "results/grain_math_response_v1/query_context_cube_r1"
OUT = ROOT / "results/grain_math_response_v1/query_context_cube_verification_r1"
BITS = tuple(itertools.product(range(2), repeat=3))
LABELS = ("mean", "Q", "R", "RQ", "H", "HQ", "HR", "HRQ")


def product_parts(x):
    """Project with J and I-J, without using the primary Fourier transform."""
    parts, energies = [], {}
    for mask, label in enumerate(LABELS):
        part = x.copy()
        for axis, bit in enumerate((4, 2, 1)):
            average = np.broadcast_to(part.mean(axis=axis, keepdims=True), part.shape)
            part = part - average if mask & bit else average.copy()
        parts.append(part)
        energies[label + "_energy"] = float(np.sum(part * part) / 8)
    reconstructed = sum(parts)
    total = float(np.sum((x - parts[0]) ** 2) / 8)
    all_energy = float(np.sum(x * x) / 8)
    stats = {
        **energies,
        **{
            k + "_share": energies[k + "_energy"] / total if total else None
            for k in LABELS[1:]
        },
        "total_centered_energy": total,
        "mean_square_energy": all_energy,
        "reconstruction_error": float(np.max(abs(x - reconstructed))),
        "energy_identity_error": abs(all_energy - sum(energies.values())),
    }
    for name, axes in (("H", (1, 2)), ("HQ", (1,)), ("HR", (2,))):
        residual = x - x.mean(axis=axes, keepdims=True)
        bound = float(np.sum(residual * residual) / 8)
        stats[name + "_only_mse_bound"] = bound
        stats[name + "_only_relative_rmse_bound"] = (
            float(np.sqrt(bound / all_energy)) if all_energy else None
        )
    return np.array([p[0, 0, 0] for p in parts]), stats


def verify(parent, out):
    """Replay all references, interventions and result tables with zero model calls."""
    if out.exists():
        raise FileExistsError("preserve previous verifier output")
    m = check(parent)
    spec = m["spec"]
    natural, fiber = ROOT / spec["natural_parent"], ROOT / spec["parent"]
    if (
        read_json(ROOT / spec["parent_verification"] / "summary.json")["status"]
        != "VERIFIED_FINITE_HM31_NOT_GLOBAL"
    ):
        raise ValueError("verified inherited corners required")
    ledger = read_json(parent / "02/artifact_hashes.json")
    for name, sha in ledger.items():
        if digest(parent / name) != sha:
            raise ValueError("frozen output changed " + name)
    design = read_json(parent / "00/design.json")
    natural_rows = read_json(natural / "00/design.json")
    fiber_rows = read_json(fiber / "00/design.json")
    expected = {
        (p, w, q, *bits)
        for p in range(4)
        for w in range(4)
        for q in spec["queries"]
        for bits in BITS
    }
    actual = {
        (
            r["pair"],
            r["wording"],
            r["query"],
            r["donor_side"],
            r["receiver_side"],
            r["query_side"],
        )
        for r in design
    }
    if (
        len(design) != 384
        or actual != expected
        or [r["unit"] for r in design] != list(range(384))
    ):
        raise ValueError("48 complete eight-corner units required")
    calls = read_json(parent / "run_status.json")
    if (calls["forward_calls"], calls["backward_calls"], calls["in_flight"]) != (
        960,
        192,
        None,
    ):
        raise ValueError("960 actual new forwards/192 backwards required")
    if calls["model_seconds"] > spec["maximum_model_seconds"] or m["new_fits"] != 0:
        raise ValueError("frozen acquisition budget exceeded")
    barrier_path = parent / "01/source_barrier.json"
    barrier = read_json(barrier_path)
    if barrier["new_fits"] != 0 or barrier["parent_source_sha256"] != digest(
        natural / "01/source_manifest.json"
    ):
        raise ValueError("checkpoint changed source fields")
    for name, sha in barrier["source_artifacts"].items():
        if digest(parent / name) != sha:
            raise ValueError("source artifact changed after checkpoint")
    errors = {
        k: 0.0
        for k in (
            "native",
            "gradient_readout",
            "action",
            "array",
            "metric",
            "decomposition",
            "contrast",
            "pooled",
            "hook",
            "same_h",
            "same_q",
            "same_delta",
        )
    }

    def compare(kind, actual_value, reported):
        if actual_value is None:
            if reported not in (None, ""):
                raise ValueError("undefined statistic fabricated")
        else:
            difference = np.max(
                abs(np.asarray(actual_value, float) - np.asarray(reported, float))
            )
            if not np.isfinite(difference):
                raise ValueError("nonfinite verification comparison")
            errors[kind] = max(errors[kind], float(difference))

    def replay(meta):
        compare(
            "native",
            score(meta["native"], m["alphabet"]["values"]),
            meta["native"]["conditional_score"],
        )

    def chronology(meta, role):
        t = datetime.fromisoformat(meta["acquired_at"])
        checkpoint = datetime.fromisoformat(barrier["frozen_at"])
        if role == "source":
            if (
                t >= checkpoint
                or meta["source_sha256"] is not None
                or t <= datetime.fromisoformat(m["frozen_before_outputs"])
            ):
                raise ValueError(
                    "source must follow protocol freeze and precede checkpoint"
                )
        elif t <= checkpoint or meta["source_sha256"] != digest(barrier_path):
            raise ValueError("held must follow source checkpoint")

    exact_matrix, exact_inverse = rational_inverse()
    matrix, inverse = np.array(exact_matrix, float), np.array(exact_inverse, float)
    fitted = {}
    for key, entry in read_json(natural / "01/source_manifest.json").items():
        for name, sha in {
            entry["file"]: entry["sha256"],
            **entry["source_hashes"],
        }.items():
            if digest(natural / name) != sha:
                raise ValueError("frozen field dependency changed")
        with np.load(natural / entry["file"], allow_pickle=False) as data:
            fitted[key] = data["covector"].copy()
    if len(fitted) != 15:
        raise ValueError("all15 unchanged fields required")
    with np.load(parent / "00/directions.npz", allow_pickle=False) as data:
        bank = {k: data[k].copy() for k in data.files}
    directions = {}
    for q in spec["queries"]:
        b = bank[q + "_b"] / np.linalg.norm(bank[q + "_b"])
        directions[q] = {
            a: np.outer(
                inverse @ ([1.0, 0, 0, 0] if a == "known_mean" else [0.0, -1, 0, 1]), b
            )
            for a in spec["arms"]
        }
    originals = {}
    for r in natural_rows:
        p = natural / "01/baselines" / f"unit_{r['unit']:03d}.json"
        if digest(p.with_suffix(".npz")) != read_json(p)["array_sha256"]:
            raise ValueError("natural state hash changed")
        with np.load(p.with_suffix(".npz"), allow_pickle=False) as data:
            originals[r["unit"]] = {"h": data["h"].copy(), "q": data["q"].copy()}
    reported = {(int(r["unit"]), r["arm"]): r for r in table(parent / "02/actions.csv")}
    reported_ends = {
        (int(r["unit"]), r["arm"], r["sign"]): r
        for r in table(parent / "02/endpoints.csv")
    }
    refs, actions, deltas = {}, {}, {}
    new_refs = new_ends = 0
    for r in design:
        old = fiber_rows[r["fiber_unit"]]
        if (
            any(r[k] != v for k, v in old.items() if k != "unit")
            or r["fiber_unit"] != r["unit"] // 2
        ):
            raise ValueError("inherited experimental address changed")
        i, j, k = r["donor_side"], r["receiver_side"], r["query_side"]
        query_row = next(
            a
            for a in fiber_rows
            if (
                a["pair"],
                a["wording"],
                a["query"],
                a["donor_side"],
                a["receiver_side"],
            )
            == (r["pair"], r["wording"], r["query"], k, k)
        )
        qnatural = natural_rows[query_row["receiver_unit"]]
        if (
            r["query_donor_unit"] != qnatural["unit"]
            or r["inherited"] != (j == k)
            or r["query_endpoint_token_id"] != r["input_ids"][-1]
            or r["input_ids"][r["question_start"] :]
            != qnatural["input_ids"][qnatural["question_start"] :]
        ):
            raise ValueError("query tokens or donor binding changed")
        role = (
            "source"
            if r["pair"] < 2
            else "validation"
            if r["pair"] == 2
            else "application"
        )
        if r["role"] != role:
            raise ValueError("source/held roles changed")
        if r["inherited"]:
            owner, unit = (
                (natural, r["donor_unit"])
                if r["diagonal"]
                else (fiber, r["fiber_unit"])
            )
            address = natural_rows[unit] if r["diagonal"] else old
        else:
            owner, unit, address = parent, r["unit"], r
        p = owner / "01/baselines" / f"unit_{unit:03d}.json"
        meta = read_json(p)
        if (
            meta["address"] != address
            or digest(p.with_suffix(".npz")) != meta["array_sha256"]
        ):
            raise ValueError("reference address/array changed")
        replay(meta)
        with np.load(p.with_suffix(".npz"), allow_pickle=False) as data:
            h, qstate, g = (
                data["h"].copy(),
                data["q"].copy(),
                data["gradient"].astype(float),
            )
            if not r["inherited"]:
                compare(
                    "same_h", data["receiver_h"], originals[r["receiver_unit"]]["h"]
                )
                compare(
                    "same_q", data["receiver_q"], originals[r["receiver_unit"]]["q"]
                )
        compare("same_h", h, originals[r["donor_unit"]]["h"])
        compare("same_q", qstate, originals[r["query_donor_unit"]]["q"])
        if not r["inherited"]:
            new_refs += 1
            chronology(meta, role)
            for v in meta["audit"].values():
                compare("hook", v, 0)
            compare(
                "gradient_readout",
                meta["gradient_readout"],
                meta["native"]["conditional_score"],
            )
        refs[r["unit"]] = (h, qstate, g, meta["native"]["conditional_score"])
        for arm in spec["arms"]:
            direction = directions[r["query"]][arm]
            intended = (
                direction
                / np.linalg.norm(direction)
                * spec["radius_relative_span_norm"]
                * np.linalg.norm(h)
            )
            plus = (h + intended.astype(np.float32)).astype(np.float32)
            minus = (h - intended.astype(np.float32)).astype(np.float32)
            delta = (plus.astype(float) - minus.astype(float)) / 2
            deltas[r["unit"], arm] = delta
            ends = {}
            for sign in ("plus", "minus"):
                ep = owner / "01/patches" / f"unit_{unit:03d}_{arm}_{sign}.json"
                end = read_json(ep)
                replay(end)
                if end["address"] != {"unit": unit, "arm": arm, "sign": sign} or end[
                    "baseline_sha256"
                ] != digest(p):
                    raise ValueError("endpoint reference address changed")
                for value in end["audit"].values():
                    compare("hook", value, 0)
                if not r["inherited"]:
                    new_ends += 1
                    chronology(end, role)
                ends[sign] = end["native"]["conditional_score"]
                if reported_ends[r["unit"], arm, sign]["inherited"] != str(
                    r["inherited"]
                ):
                    raise ValueError("endpoint inheritance flag changed")
                for name, value in end["native"].items():
                    if isinstance(value, (int, float)):
                        compare(
                            "native", value, reported_ends[r["unit"], arm, sign][name]
                        )
            moments = matrix @ delta
            a = {
                "odd_action": (ends["plus"] - ends["minus"]) / 2,
                "even_change": (ends["plus"] + ends["minus"]) / 2 - refs[r["unit"]][3],
                "plus_change": ends["plus"] - refs[r["unit"]][3],
                "minus_change": ends["minus"] - refs[r["unit"]][3],
                "physical_delta_norm": float(np.sqrt(np.sum(delta * delta))),
                "global_source": float(
                    np.sum(fitted[r["query"] + "_global"] * moments)
                ),
                "wording_source": float(
                    np.sum(fitted[r["query"] + f"_wording{r['wording']}"] * moments)
                ),
                "local_derivative": float(np.sum(g * delta)),
            }
            for name, value in a.items():
                compare("action", value, reported[r["unit"], arm][name])
            actions[r["unit"], arm] = {**r, "arm": arm, **a}
    if (new_refs, new_ends, len(reported), len(reported_ends)) != (192, 768, 768, 1536):
        raise ValueError("actual new/inherited coverage wrong")
    rm = lambda a: float(np.sqrt(np.mean(np.asarray(a, float) ** 2)))
    metrics = table(parent / "02/metrics.csv")
    if len(metrics) != 36:
        raise ValueError("36cold metrics required")
    for r in metrics:
        selected = [
            a
            for a in actions.values()
            if (a["role"], a["query"], a["arm"], a["inherited"])
            == (r["role"], r["query"], r["arm"], r["inherited"] == "True")
        ]
        y, d = (
            np.array([a["odd_action"] for a in selected]),
            np.array([a["physical_delta_norm"] for a in selected]),
        )
        values = {
            "cells": len(selected),
            "contexts": len({a["context"] for a in selected}),
            "wordings": len({a["wording"] for a in selected}),
            "native_action_rms": rm(y),
            "native_unit_action_rms": rm(y / d),
            "even_change_rms": rm([a["even_change"] for a in selected]),
        }
        for method in ("global_source", "wording_source", "local_derivative"):
            prediction = np.array([a[method] for a in selected])
            values.update(
                {
                    method + "_rmse": rm(prediction - y),
                    method + "_unit_rmse": rm((prediction - y) / d),
                    method + "_prediction_rms": rm(prediction),
                    method + "_signs": int(sum(prediction * y > 0)),
                }
            )
        for name, value in values.items():
            compare("metric", value, r[name])
    decomp, ctable, pooled = (
        table(parent / "02/cube_decomposition.csv"),
        table(parent / "02/cube_contrasts.csv"),
        table(parent / "02/pooled_decomposition.csv"),
    )
    if (len(decomp), len(ctable), len(pooled)) != (576, 5184, 108):
        raise ValueError("complete decomposition/contrast/pooled tables required")
    cindex = {}
    for c in ctable:
        key = (int(c["pair"]), int(c["wording"]), c["query"], c["space"], c["arm"])
        cindex.setdefault(key, []).append(c)
    alternate = []
    with np.load(parent / "02/cube_arrays.npz", allow_pickle=False) as arrays:
        for r in decomp:
            p, w, q, space, arm = (
                int(r["pair"]),
                int(r["wording"]),
                r["query"],
                r["space"],
                r["arm"],
            )
            by = {
                (s["donor_side"], s["receiver_side"], s["query_side"]): s
                for s in design
                if (s["pair"], s["wording"], s["query"]) == (p, w, q)
            }
            g = np.array([refs[by[s]["unit"]][2] for s in BITS]).reshape(2, 2, 2, 4, -1)
            if space == "gradient":
                x = g
            elif space == "covector":
                x = np.einsum("ab,ijkbd->ijkad", inverse.T, g)
            elif space == "task_gradient":
                x = np.array(
                    [
                        [
                            float(
                                np.sum(
                                    refs[by[s]["unit"]][2]
                                    * directions[q][a]
                                    / np.linalg.norm(directions[q][a])
                                )
                            )
                            for a in spec["arms"]
                        ]
                        for s in BITS
                    ]
                ).reshape(2, 2, 2, 2)
            elif space == "native_level":
                x = np.array([refs[by[s]["unit"]][3] for s in BITS]).reshape(2, 2, 2)
            else:
                method = (
                    "wording_source"
                    if space == "cold_unit_action"
                    else "local_derivative"
                    if space == "local_unit_action"
                    else "odd_action"
                )
                x = np.array(
                    [
                        actions[by[s]["unit"], arm][method]
                        / (
                            1
                            if space == "raw_action"
                            else actions[by[s]["unit"], arm]["physical_delta_norm"]
                        )
                        for s in BITS
                    ]
                ).reshape(2, 2, 2)
            key = f"p{p}_w{w}_{q}_{space}_{arm}"
            compare("array", x, arrays[key])
            coefficients, stats = product_parts(x)
            compare("array", coefficients, arrays[key + "_coefficients"])
            for name, value in stats.items():
                compare("decomposition", value, r[name])
            alternate.append(
                {"role": r["role"], "query": q, "space": space, "arm": arm, **stats}
            )
            for i in range(2):
                for a in spec["arms"]:
                    for j, k in itertools.product(range(2), repeat=2):
                        compare(
                            "same_delta",
                            deltas[by[i, j, k]["unit"], a],
                            deltas[by[i, 0, 0]["unit"], a],
                        )
            selected = cindex.get((p, w, q, space, arm), [])
            if (
                space
                in (
                    "gradient",
                    "task_gradient",
                    "native_level",
                    "raw_action",
                    "unit_action",
                    "cold_unit_action",
                )
                and len(selected) != 12
            ):
                raise ValueError("all12 single-axis contrasts required")
            for c in selected:
                axis = ("H", "R", "Q").index(c["vary"])
                indices = [0, 0, 0]
                indices[("H", "R", "Q").index(c["fixed0_axis"])] = int(c["fixed0_side"])
                indices[("H", "R", "Q").index(c["fixed1_axis"])] = int(c["fixed1_side"])
                left = x[tuple(indices)]
                indices[axis] = 1
                right = x[tuple(indices)]
                delta = left - right
                norm = float(np.sqrt(np.sum(delta * delta)))
                ref = float(np.sqrt((np.sum(left * left) + np.sum(right * right)) / 2))
                for name, value in {
                    "difference_norm": norm,
                    "reference_rms_norm": ref,
                    "relative_difference": norm / ref if ref else None,
                    "signed_difference": float(delta) if delta.ndim == 0 else None,
                }.items():
                    compare("contrast", value, c[name])
    for r in pooled:
        selected = [
            a
            for a in alternate
            if all(a[k] == r[k] for k in ("role", "query", "space", "arm"))
        ]
        values = {
            k: float(np.mean([a[k] for a in selected]))
            for k in selected[0]
            if k.endswith(("_energy", "_mse_bound"))
        }
        for label in LABELS[1:]:
            values[label + "_share"] = (
                values[label + "_energy"] / values["total_centered_energy"]
                if values["total_centered_energy"]
                else None
            )
        for label in ("H", "HQ", "HR"):
            values[label + "_only_relative_rmse_bound"] = (
                float(
                    np.sqrt(
                        values[label + "_only_mse_bound"] / values["mean_square_energy"]
                    )
                )
                if values["mean_square_energy"]
                else None
            )
        values["cubes"] = len(selected)
        for name, value in values.items():
            compare("pooled", value, r[name])
    if any(
        v > (1e-6 if k == "gradient_readout" else 1e-10) for k, v in errors.items()
    ) or any(errors[k] for k in ("hook", "same_h", "same_q", "same_delta")):
        raise ValueError("alternate reconstruction mismatch " + str(errors))
    out.mkdir(parents=True)
    write_json(
        out / "summary.json",
        {
            "status": "VERIFIED_FINITE_HM32_NOT_GLOBAL",
            "counts": read_json(parent / "02/summary.json")["counts"],
            "maximum_errors": errors,
            "artifact_hashes_checked": len(ledger),
            "inherited_hashes_checked": len(m["files"]),
            "source_barrier_sha256": digest(barrier_path),
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "independence": "same-agent alternate score/rational/product-projection/conditional-variance formulas, not third-party or psychological validation",
            "new_model_calls": 0,
            "new_fits": 0,
        },
    )
    print(read_json(out / "summary.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    verify(args.parent, args.out)
