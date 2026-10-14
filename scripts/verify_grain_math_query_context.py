#!/usr/bin/env python3
"""HM33 alternate conditional projections, rational units and complete replay."""

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
from scripts import run_grain_math_query_cube as OLD
from scripts.run_grain_math_position_support import (
    array_unit,
    baseline_path,
    check,
    patch_path,
)
from scripts.run_grain_math_response import write_json
from scripts.verify_grain_math_crossed_field import table
from scripts.verify_grain_math_position_support import rational_inverse, score

PARENT = ROOT / "results/grain_math_response_v1/early_query_context_localization_r1"
OUT = ROOT / "results/grain_math_response_v1/early_query_context_verification_r1"
BITS = tuple(itertools.product(range(2), repeat=4))
AXES = ("H", "B", "P", "Q")
LABELS = tuple(
    "".join(n for n, b in zip(AXES, (8, 4, 2, 1), strict=True) if mask & b) or "mean"
    for mask in range(16)
)


def product_parts(x):
    """Alternate J/(I-J) projections and conditional means, no character transform."""
    parts, energies = [], {}
    for mask, label in enumerate(LABELS):
        part = x.copy()
        for axis, bit in enumerate((8, 4, 2, 1)):
            mean = np.broadcast_to(part.mean(axis=axis, keepdims=True), part.shape)
            part = part - mean if mask & bit else mean.copy()
        parts.append(part)
        energies[label + "_energy"] = float(np.sum(part * part) / 16)
    total = float(np.sum((x - parts[0]) ** 2) / 16)
    all_energy = float(np.sum(x * x) / 16)
    stats = {
        **energies,
        **{
            label + "_share": energies[label + "_energy"] / total if total else None
            for label in LABELS[1:]
        },
        "total_centered_energy": total,
        "mean_square_energy": all_energy,
        "reconstruction_error": float(np.max(abs(x - sum(parts)))),
        "energy_identity_error": abs(all_energy - sum(energies.values())),
    }
    for name, axes in (("HQ", (1, 2)), ("HPQ", (1,)), ("HBQ", (2,))):
        residual = x - x.mean(axis=axes, keepdims=True)
        bound = float(np.sum(residual * residual) / 16)
        stats[name + "_only_mse_bound"] = bound
        stats[name + "_only_relative_rmse_bound"] = (
            float(np.sqrt(bound / all_energy)) if all_energy else None
        )
    return np.array([p[0, 0, 0, 0] for p in parts]), stats


def verify(parent, out):
    """Reconstruct every table and old diagonal without new model calls or fits."""
    if out.exists():
        raise FileExistsError("preserve previous verification")
    m = check(parent)
    spec = m["spec"]
    old, natural = ROOT / spec["parent"], ROOT / spec["natural_parent"]
    om = read_json(old / "00/manifest.json")
    old_rows = read_json(old / "00/design.json")
    natural_rows = read_json(natural / "00/design.json")
    ledger = read_json(parent / "02/artifact_hashes.json")
    for name, sha in ledger.items():
        if digest(parent / name) != sha:
            raise ValueError("frozen output changed " + name)
    design = read_json(parent / "00/design.json")
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
            r["early_query_side"],
            r["query_side"],
        )
        for r in design
    }
    if (
        len(design) != 768
        or expected != actual
        or [r["unit"] for r in design] != list(range(768))
    ):
        raise ValueError("48 complete16-corner designs required")
    calls = read_json(parent / "run_status.json")
    if (
        (calls["forward_calls"], calls["backward_calls"], calls["in_flight"])
        != (2016, 384, None)
        or calls["model_seconds"] > spec["maximum_model_seconds"]
        or m["new_fits"]
    ):
        raise ValueError("actual acquisition budget/accounting mismatch")
    barrier_path = parent / "01/source_barrier.json"
    barrier = read_json(barrier_path)
    if barrier["new_fits"] or barrier["parent_source_sha256"] != digest(
        natural / "01/source_manifest.json"
    ):
        raise ValueError("unchanged source field checkpoint required")
    for name, sha in barrier["source_artifacts"].items():
        if digest(parent / name) != sha:
            raise ValueError("source artifact changed after checkpoint")
    errors = {
        k: 0.0
        for k in (
            "native",
            "gradient_readout",
            "same_h",
            "same_p",
            "same_q",
            "same_delta",
            "hook",
            "action",
            "metric",
            "array",
            "decomposition",
            "contrast",
            "pooled",
            "old_restriction",
            "old_alias",
        )
    }

    def compare(kind, actual_value, reported):
        if actual_value is None:
            if reported not in (None, ""):
                raise ValueError("undefined statistic fabricated")
            return
        delta = np.max(
            abs(np.asarray(actual_value, float) - np.asarray(reported, float))
        )
        if not np.isfinite(delta):
            raise ValueError("nonfinite replay")
        errors[kind] = max(errors[kind], float(delta))

    def replay(meta):
        compare(
            "native",
            score(meta["native"], m["alphabet"]["values"]),
            meta["native"]["conditional_score"],
        )

    def chronology(meta, role):
        t, freeze, start = (
            datetime.fromisoformat(v)
            for v in (
                meta["acquired_at"],
                barrier["frozen_at"],
                m["frozen_before_outputs"],
            )
        )
        if role == "source":
            if not start < t < freeze or meta["source_sha256"] is not None:
                raise ValueError("source chronology violated")
        elif t <= freeze or meta["source_sha256"] != digest(barrier_path):
            raise ValueError("held chronology violated")

    captures = {}
    for row in natural_rows:
        p = parent / "01/captures" / f"unit_{row['unit']:03d}.json"
        meta = array_unit(p)
        pp = list(range(row["question_start"], len(row["input_ids"]) - 1))
        if (
            meta["address"] != row
            or meta["early_query_positions"] != pp
            or meta["early_query_token_ids"] != [row["input_ids"][i] for i in pp]
        ):
            raise ValueError("natural P token addresses changed")
        chronology(meta, row["role"])
        replay(meta)
        old_native = read_json(baseline_path(natural, row["unit"]))["native"]
        if meta["native"] != old_native:
            raise ValueError("original natural readout changed")
        with (
            np.load(p.with_suffix(".npz"), allow_pickle=False) as data,
            np.load(
                baseline_path(natural, row["unit"]).with_suffix(".npz"),
                allow_pickle=False,
            ) as original,
        ):
            captures[row["unit"]] = {k: data[k].copy() for k in ("h", "p", "q")}
            for k in ("h", "q"):
                compare("same_" + k, data[k], original[k])
            if (
                data["p"].shape != (len(pp), data["h"].shape[-1])
                or not np.isfinite(data["p"]).all()
            ):
                raise ValueError("finite natural P capture shape required")
    if len(captures) != 96:
        raise ValueError("all96 original input captures required")
    exact_matrix, exact_inverse = rational_inverse()
    matrix, inverse = np.array(exact_matrix, float), np.array(exact_inverse, float)
    fitted = {}
    for key, entry in read_json(natural / "01/source_manifest.json").items():
        for name, sha in {
            entry["file"]: entry["sha256"],
            **entry["source_hashes"],
        }.items():
            if digest(natural / name) != sha:
                raise ValueError("field ancestry changed")
        with np.load(natural / entry["file"], allow_pickle=False) as data:
            fitted[key] = data["covector"].copy()
    if len(fitted) != 15:
        raise ValueError("unchanged15 fields required")
    with np.load(parent / "00/directions.npz", allow_pickle=False) as data:
        bank = {k: data[k].copy() for k in data.files}
    directions = {
        q: {
            a: np.outer(
                inverse @ ([1.0, 0, 0, 0] if a == "known_mean" else [0.0, -1, 0, 1]),
                bank[q + "_b"] / np.linalg.norm(bank[q + "_b"]),
            )
            for a in spec["arms"]
        }
        for q in spec["queries"]
    }
    reported = {(int(r["unit"]), r["arm"]): r for r in table(parent / "02/actions.csv")}
    reported_ends = {
        (int(r["unit"]), r["arm"], r["sign"]): r
        for r in table(parent / "02/endpoints.csv")
    }
    refs, actions, deltas = {}, {}, {}
    new_refs = new_ends = 0
    for r in design:
        original = old_rows[r["cube_unit"]]
        if (
            r["cube_unit"] != r["unit"] // 2
            or any(
                r[k] != v for k, v in original.items() if k not in ("unit", "inherited")
            )
            or r["cube_inherited"] != original["inherited"]
        ):
            raise ValueError("original cube address changed")
        i, j, l, k = (
            r[key]
            for key in ("donor_side", "receiver_side", "early_query_side", "query_side")
        )
        donor = next(
            a
            for a in old_rows
            if (
                a["pair"],
                a["wording"],
                a["query"],
                a["donor_side"],
                a["receiver_side"],
                a["query_side"],
            )
            == (r["pair"], r["wording"], r["query"], l, l, l)
        )
        pd = natural_rows[donor["receiver_unit"]]
        pp = list(range(r["question_start"], len(r["input_ids"]) - 1))
        if (
            r["inherited"] != (j == l)
            or r["early_query_donor_unit"] != pd["unit"]
            or r["early_query_positions"] != pp
            or r["early_query_token_ids"] != pd["input_ids"][pd["question_start"] : -1]
            or r["input_ids"][-1] != pd["input_ids"][-1]
        ):
            raise ValueError("exact P/Q sequence/donor binding changed")
        role = (
            "source"
            if r["pair"] < 2
            else "validation"
            if r["pair"] == 2
            else "application"
        )
        if r["role"] != role or pd["role"] != role:
            raise ValueError("source/held roles changed")
        if r["inherited"]:
            path = OLD.ref_path(old, original, om)
            end_path = lambda a, s, address=original: OLD.end_path(
                old, address, om, a, s
            )
        else:
            path = baseline_path(parent, r["unit"])
            end_path = lambda a, s, unit=r["unit"]: patch_path(parent, unit, a, s)
        meta = array_unit(path)
        replay(meta)
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
            h, qstate, g = (
                data["h"].copy(),
                data["q"].copy(),
                data["gradient"].astype(float),
            )
            for key, unit in (("h", r["donor_unit"]), ("q", r["query_donor_unit"])):
                compare("same_" + key, data[key], captures[unit][key])
            if not r["inherited"]:
                for key in ("h", "p", "q"):
                    compare(
                        "same_" + key,
                        data["receiver_" + key],
                        captures[r["receiver_unit"]][key],
                    )
                compare("same_p", data["p"], captures[r["early_query_donor_unit"]]["p"])
        if not r["inherited"]:
            if meta["address"] != r:
                raise ValueError("new reference address changed")
            new_refs += 1
            chronology(meta, role)
            compare(
                "gradient_readout",
                meta["gradient_readout"],
                meta["native"]["conditional_score"],
            )
        for v in meta.get("audit", {}).values():
            compare("hook", v, 0)
        refs[r["unit"]] = (h, qstate, g, meta["native"]["conditional_score"])
        for arm in spec["arms"]:
            direction = directions[r["query"]][arm]
            step = (
                direction
                / np.linalg.norm(direction)
                * spec["radius_relative_span_norm"]
                * np.linalg.norm(h)
            ).astype(np.float32)
            delta = (
                (h + step).astype(np.float32).astype(float)
                - (h - step).astype(np.float32).astype(float)
            ) / 2
            deltas[r["unit"], arm] = delta
            ends = {}
            for sign in ("plus", "minus"):
                ep = end_path(arm, sign)
                end = read_json(ep)
                replay(end)
                if end["baseline_sha256"] != digest(path) or end["address"] != {
                    "unit": meta["address"]["unit"],
                    "arm": arm,
                    "sign": sign,
                }:
                    raise ValueError("endpoint binding changed")
                if not r["inherited"]:
                    new_ends += 1
                    chronology(end, role)
                for v in end["audit"].values():
                    compare("hook", v, 0)
                ends[sign] = end["native"]["conditional_score"]
                report = reported_ends[r["unit"], arm, sign]
                if report["inherited"] != str(r["inherited"]):
                    raise ValueError("inheritance flag changed")
                for name, value in end["native"].items():
                    if isinstance(value, bool):
                        if report[name] != str(value):
                            raise ValueError("native boolean changed")
                    elif isinstance(value, (int, float)):
                        compare("native", value, report[name])
            moments = matrix @ delta
            y = refs[r["unit"]][3]
            a = {
                "odd_action": (ends["plus"] - ends["minus"]) / 2,
                "even_change": (ends["plus"] + ends["minus"]) / 2 - y,
                "plus_change": ends["plus"] - y,
                "minus_change": ends["minus"] - y,
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
    if (new_refs, new_ends, len(reported), len(reported_ends)) != (
        384,
        1536,
        1536,
        3072,
    ):
        raise ValueError("complete new/inherited acquisition required")
    rm = lambda a: float(np.sqrt(np.mean(np.asarray(a, float) ** 2)))
    metrics = table(parent / "02/metrics.csv")
    if len(metrics) != 36:
        raise ValueError("all36 cold metrics required")
    for r in metrics:
        rows = [
            a
            for a in actions.values()
            if (a["role"], a["query"], a["arm"], a["inherited"])
            == (r["role"], r["query"], r["arm"], r["inherited"] == "True")
        ]
        y = np.array([a["odd_action"] for a in rows])
        d = np.array([a["physical_delta_norm"] for a in rows])
        values = {
            "cells": len(rows),
            "contexts": len({a["context"] for a in rows}),
            "wordings": len({a["wording"] for a in rows}),
            "native_action_rms": rm(y),
            "native_unit_action_rms": rm(y / d),
            "even_change_rms": rm([a["even_change"] for a in rows]),
        }
        for method in ("global_source", "wording_source", "local_derivative"):
            pred = np.array([a[method] for a in rows])
            values.update(
                {
                    method + "_rmse": rm(pred - y),
                    method + "_unit_rmse": rm((pred - y) / d),
                    method + "_prediction_rms": rm(pred),
                    method + "_signs": int(sum(pred * y > 0)),
                }
            )
        for name, value in values.items():
            compare("metric", value, r[name])
    decomp, contrasts, pooled = (
        table(parent / ("02/" + n + ".csv"))
        for n in ("cube_decomposition", "cube_contrasts", "pooled_decomposition")
    )
    if (len(decomp), len(contrasts), len(pooled)) != (576, 13824, 108):
        raise ValueError("all decomposition/contrast/pooled rows required")
    cindex = {}
    for c in contrasts:
        cindex.setdefault(
            (int(c["pair"]), int(c["wording"]), c["query"], c["space"], c["arm"]), []
        ).append(c)
    alternate = []
    with (
        np.load(parent / "02/cube_arrays.npz", allow_pickle=False) as arrays,
        np.load(old / "02/cube_arrays.npz", allow_pickle=False) as old_arrays,
    ):
        for r in decomp:
            pair, w, q, space, arm = (
                int(r["pair"]),
                int(r["wording"]),
                r["query"],
                r["space"],
                r["arm"],
            )
            by = {
                (
                    s["donor_side"],
                    s["receiver_side"],
                    s["early_query_side"],
                    s["query_side"],
                ): s
                for s in design
                if (s["pair"], s["wording"], s["query"]) == (pair, w, q)
            }
            g = np.array([refs[by[s]["unit"]][2] for s in BITS]).reshape(
                2, 2, 2, 2, 4, -1
            )
            if space == "gradient":
                x = g
            elif space == "covector":
                x = np.einsum("ab,ijklbd->ijklad", inverse.T, g)
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
                ).reshape(2, 2, 2, 2, 2)
            elif space == "native_level":
                x = np.array([refs[by[s]["unit"]][3] for s in BITS]).reshape(2, 2, 2, 2)
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
                ).reshape(2, 2, 2, 2)
            key = f"p{pair}_w{w}_{q}_{space}_{arm}"
            compare("array", x, arrays[key])
            coefficients, stats = product_parts(x)
            compare("array", coefficients, arrays[key + "_coefficients"])
            for name, value in stats.items():
                compare("decomposition", value, r[name])
            alternate.append(
                {"role": r["role"], "query": q, "space": space, "arm": arm, **stats}
            )
            restricted = np.array(
                [
                    [[x[i, j, j, k] for k in range(2)] for j in range(2)]
                    for i in range(2)
                ]
            )
            compare("old_restriction", restricted, old_arrays[key])
            collapsed = np.array(
                [
                    coefficients[((mask & 4) << 1) | ((mask & 2) << 1) | (mask & 1)]
                    + coefficients[
                        (((mask & 4) << 1) | ((mask & 2) << 1) | (mask & 1)) ^ 6
                    ]
                    for mask in range(8)
                ]
            )
            compare("old_alias", collapsed, old_arrays[key + "_coefficients"])
            for i in range(2):
                for a in spec["arms"]:
                    for j, l, k in itertools.product(range(2), repeat=3):
                        compare(
                            "same_delta",
                            deltas[by[i, j, l, k]["unit"], a],
                            deltas[by[i, 0, 0, 0]["unit"], a],
                        )
            selected = cindex.get((pair, w, q, space, arm), [])
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
                and len(selected) != 32
            ):
                raise ValueError("complete32 single-carrier contrasts required")
            for c in selected:
                index = [0] * 4
                for n in range(3):
                    index[AXES.index(c[f"fixed{n}_axis"])] = int(c[f"fixed{n}_side"])
                left = x[tuple(index)]
                index[AXES.index(c["vary"])] = 1
                right = x[tuple(index)]
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
        rows = [
            a
            for a in alternate
            if all(a[k] == r[k] for k in ("role", "query", "space", "arm"))
        ]
        values = {
            k: float(np.mean([a[k] for a in rows]))
            for k in rows[0]
            if k.endswith(("_energy", "_mse_bound"))
        }
        for label in LABELS[1:]:
            values[label + "_share"] = (
                values[label + "_energy"] / values["total_centered_energy"]
                if values["total_centered_energy"]
                else None
            )
        for name in ("HQ", "HPQ", "HBQ"):
            values[name + "_only_relative_rmse_bound"] = (
                float(
                    np.sqrt(
                        values[name + "_only_mse_bound"] / values["mean_square_energy"]
                    )
                )
                if values["mean_square_energy"]
                else None
            )
        values["cubes"] = len(rows)
        for name, value in values.items():
            compare("pooled", value, r[name])
    if any(
        v > (1e-6 if k == "gradient_readout" else 1e-10) for k, v in errors.items()
    ) or any(errors[k] for k in ("same_h", "same_p", "same_q", "same_delta", "hook")):
        raise ValueError("alternate replay mismatch " + str(errors))
    out.mkdir(parents=True)
    write_json(
        out / "summary.json",
        {
            "status": "VERIFIED_FINITE_HM33_NOT_GLOBAL",
            "counts": read_json(parent / "02/summary.json")["counts"],
            "maximum_errors": errors,
            "artifact_hashes_checked": len(ledger),
            "inherited_hashes_checked": len(m["files"]),
            "source_barrier_sha256": digest(barrier_path),
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "independence": "same-agent alternate score/rational/product-projection/conditional-mean and old-diagonal checks; not third-party or psychological validation",
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
