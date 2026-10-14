#!/usr/bin/env python3
"""HM104: actual qualified text corners and their empirical response charts."""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_history_routing as R
from grain_research.construction import math_text_response as M
from grain_research.construction.math_author_evidence_merge import registered_queries
from grain_research.construction.math_response_connection import (
    AttentionResponse,
    rectangle,
)
from grain_research.vector_mechanisms.navigation import MODEL, write_json
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts.run_grain_history_routing_hm101 import check as source_check
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_text_response_hm104.json"
CODE = (
    "configs/grain_text_response_hm104.json",
    "docs/GRAIN_TEXT_SUPPORTED_RESPONSE_PLAN_HM104.md",
    "grain_research/construction/math_text_response.py",
    "grain_research/construction/math_response_connection.py",
    "grain_research/construction/math_history_routing.py",
    "grain_research/construction/math_author_evidence_merge.py",
    "grain_research/vector_mechanisms/prefix.py",
    "scripts/run_grain_text_response_hm104.py",
    "scripts/run_grain_math_span_binding.py",
    "scripts/run_grain_upstream_routes.py",
    "tests/test_grain_math_text_response.py",
)


def stat(values):
    """Preserve signed extrema and RMS, including zeros and failures elsewhere."""
    a = np.asarray(values, float)
    return (
        {
            "n": int(a.size),
            "min": float(a.min()),
            "max": float(a.max()),
            "mean": float(a.mean()),
            "rms": float(np.sqrt(np.mean(a**2))),
        }
        if a.size
        else {"n": 0}
    )


def design():
    """Enumerate every required actual text corner before loading model outcomes."""
    yield -1, 1, 1
    for axis, level in itertools.product(range(6), (0, 2)):
        yield axis, level, 1
    for owner in (0, 2):
        yield -1, 1, owner
    for axis, level, owner in itertools.product(range(6), (0, 2), (0, 2)):
        yield axis, level, owner


def check(out):
    """Authenticate this finite protocol, source reader and public inputs."""
    manifest = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, manifest["code"] | manifest["source"])
    H.verify_hashes(out, manifest["inputs"])
    return manifest


def prepare(out, spec):
    """Fix text support, current token positions and source queries before capture."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve old and partial experiments")
    source = ROOT / spec["source"]
    source_check(source)
    H.verify_hashes(source, H.load(source / "01/raw_seal.json"))
    if (
        H.load(source / "verification.json")["status"]
        != "VERIFIED_FINITE_HM101_NOT_GLOBAL"
    ):
        raise ValueError("Verified source prefix calibration required")
    bundle = H.load(source / "00/bundle.json")
    if bundle["targets"] != spec["targets"]:
        raise ValueError("Source target definitions changed")
    old, new = [
        H.load(ROOT / spec[name])
        for name in ("source_material", "application_material")
    ]
    banks = [
        {t: old["behaviours"][t][0] for t in spec["targets"]},
        new["behaviours"],
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    system = H.load(ROOT / spec["system_material"])["system"]
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": "__GRAIN_BODY__"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    prefix = tokenizer.encode(
        rendered.split("__GRAIN_BODY__")[0], add_special_tokens=False
    )

    def tokens(text):
        """Encode each fixed semantic segment without future query tokens."""
        return tokenizer.encode(text, add_special_tokens=False)

    histories = []
    for bank, (axis, level, owner) in itertools.product(range(2), list(design())):
        phrases = [
            banks[bank][t][level if i == axis else 1]
            for i, t in enumerate(spec["targets"])
        ]
        text = "\n".join(
            f"Earlier entry E{i + 1}: the person {p}." for i, p in enumerate(phrases)
        )
        text += "\n" + spec["owners"][owner]
        histories.append(
            {
                "bank": bank,
                "axis": axis,
                "level": level,
                "owner": owner,
                "text": text,
                "tokens": tokens(text),
            }
        )
    width = max(len(h["tokens"]) for h in histories)
    rows = []
    for bank, bridge, family in itertools.product(range(2), range(2), range(6)):
        bt = tokens("\n" + spec["bridges"][bridge] + "\nCurrent entry: Person A")
        current_text = (
            " " + banks[bank][spec["targets"][family]][spec["current_level"]] + "."
        )
        ct = tokens(current_text)
        current = list(
            range(
                len(prefix) + width + len(bt), len(prefix) + width + len(bt) + len(ct)
            )
        )
        invariant = None
        for h in (a for a in histories if a["bank"] == bank):
            padding = list(range(len(prefix) + len(h["tokens"]), len(prefix) + width))
            ids = (
                prefix
                + h["tokens"]
                + [tokenizer.pad_token_id] * (width - len(h["tokens"]))
                + bt
                + ct
            )
            history = list(range(len(prefix), len(prefix) + width))
            current_set, history_set = set(current), set(history)
            relay = [i for i in range(len(ids)) if i not in current_set | history_set]
            identity = [ids[i] for i in current], current
            if invariant is not None and identity != invariant:
                raise ValueError("Actual current token/position invariance failed")
            invariant = identity
            rows.append(
                {
                    "unit": len(rows),
                    "bank": bank,
                    "bridge": bridge,
                    "family": family,
                    "axis": h["axis"],
                    "level": h["level"],
                    "owner": h["owner"],
                    "history_text": h["text"],
                    "current_text": current_text,
                    "input_ids": ids,
                    "history": history,
                    "relay": relay,
                    "current": current,
                    "padding": padding,
                }
            )
    if len(rows) != spec["max_forwards"] or len(
        {
            (r["bank"], r["bridge"], r["family"], r["axis"], r["level"], r["owner"])
            for r in rows
        }
    ) != len(rows):
        raise ValueError("Finite text support is incomplete or duplicated")
    (out / "00").mkdir(parents=True)
    (out / "01/units").mkdir(parents=True)
    H.write(out / "00/rows.json", rows)
    reader = {
        "targets": bundle["targets"],
        "directions": bundle["directions"],
        "positions": {
            str(e): registered_queries(bundle, e).tolist()
            for e in sorted({max(r["current"]) for r in rows})
        },
    }
    H.write(out / "00/reader.json", reader)
    sources = [source / "00/bundle.json", source / "verification.json"]
    sources += [
        ROOT / spec[name]
        for name in ("source_material", "application_material", "system_material")
    ]
    sources += [MODEL / "config.json", MODEL / "tokenizer.json"]
    manifest = {
        "packet": "HM104",
        "spec": spec,
        "code": {name: H.sha(ROOT / name) for name in CODE},
        "source": {str(p.relative_to(ROOT)): H.sha(p) for p in sources},
        "inputs": {
            f"00/{name}.json": H.sha(out / f"00/{name}.json")
            for name in ("rows", "reader")
        },
        "history_width": width,
        "actual_forwards_planned": len(rows),
        "reader_frozen_before_capture": True,
        "new_fit": False,
        "natural_continuous_text_manifold": False,
    }
    H.write(out / "00/manifest.json", manifest)
    print(
        "PREPARED HM104",
        {
            "forwards": len(rows),
            "history_width": width,
            "max_tokens": max(len(r["input_ids"]) for r in rows),
        },
        flush=True,
    )


def raw_unit(out, row):
    """Read one authenticated actual capture and its scoped execution trace."""
    path = out / f"01/units/unit_{row['unit']:04d}"
    meta = H.load(path.with_suffix(".json"))
    if H.sha(path.with_suffix(".npz")) != meta["sha256"]:
        raise ValueError("Capture checksum mismatch")
    trace = meta["trace"]
    if (
        trace["completed_blocks"] != list(range(17))
        or trace["forbidden_calls"]
        or trace["masked_blocks"] != list(range(17))
    ):
        raise ValueError("Unexpected actual capture trace")
    expected = R.route_mask(
        len(row["input_ids"]),
        row["history"],
        row["relay"],
        row["current"],
        row["padding"],
        1,
        1,
    )
    import hashlib

    if hashlib.sha256(expected.tobytes()).hexdigest() != trace["permission_sha256"]:
        raise ValueError("Actual permission mask changed")
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
        return {n: np.asarray(z[n], float) for n in ("k", "v")}


def collect(out):
    """Run the fixed local prefix battery with resumable per-unit checkpoints."""
    manifest = check(out)
    if (out / "01/raw_seal.json").exists():
        raise FileExistsError("Finished capture must not be replayed")
    rows = H.load(out / "00/rows.json")
    status_path = out / "01/status.json"
    status = (
        H.load(status_path)
        if status_path.exists()
        else {"completed": 0, "model_seconds": 0.0, "state": "RUNNING"}
    )
    observer = load_observer()
    with collector_guard(out):
        for row in rows:
            path = out / f"01/units/unit_{row['unit']:04d}"
            if path.with_suffix(".json").exists():
                raw_unit(out, row)
                continue
            if status["model_seconds"] >= manifest["spec"]["max_model_seconds"]:
                raise TimeoutError("Finite model budget reached; checkpoint retained")
            start = time.monotonic()
            raw, trace = R.capture_routed_prefix(observer, row, 1, 1)
            seconds = time.monotonic() - start
            with path.with_suffix(".npz").open("xb") as stream:
                np.savez_compressed(stream, **raw)
            H.write(
                path.with_suffix(".json"),
                {
                    "unit": row["unit"],
                    "trace": trace,
                    "seconds": seconds,
                    "sha256": H.sha(path.with_suffix(".npz")),
                },
            )
            status.update(
                completed=row["unit"] + 1,
                model_seconds=status["model_seconds"] + seconds,
            )
            write_json(status_path, status)
            if status["completed"] % 78 == 0:
                print("HM104", status, flush=True)
        status.update(state="COLLECTED", actual_forwards=len(rows))
        write_json(status_path, status)
        H.write(out / "01/raw_seal.json", H.seals(out, out / "01/units"))
    print("COLLECTED HM104", status, flush=True)


def group_data(out, rows, reader):
    """Align actual K/V arrays on a fixed current-token carrier, not pooled text."""
    data = {}
    identity = None
    for row in rows:
        key = row["axis"], row["level"], row["owner"]
        actual = raw_unit(out, row)
        ids = [row["input_ids"][p] for p in row["current"]], row["current"]
        if identity is not None and ids != identity:
            raise ValueError("Current tokens changed within a carrier context")
        identity = ids
        data[key] = actual
    q = np.asarray(reader["positions"][str(max(rows[0]["current"]))])
    d = np.asarray(reader["directions"])
    origin = data[-1, 1, 1]
    ambient = AttentionResponse(origin["k"], origin["v"], q, d)
    vectors = {
        key: np.concatenate([value["k"].ravel(), value["v"].ravel()])
        for key, value in data.items()
    }
    readings = {key: ambient.value_jacobian(x)[0] for key, x in vectors.items()}
    marginal = [vectors[i, 2, 1] - vectors[i, 0, 1] for i in range(6)]
    marginal.append(vectors[-1, 1, 2] - vectors[-1, 1, 0])
    mixed = [
        M.mixed_contrast(
            vectors[i, 2, 2], vectors[i, 0, 2], vectors[i, 2, 0], vectors[i, 0, 0]
        )
        for i in range(6)
    ]
    return (
        ambient,
        vectors,
        readings,
        np.stack(marginal, axis=1),
        np.stack(mixed, axis=1),
        q,
        d,
    )


def analyze(out):
    """Compare actual finite text interactions before any secondary numerical loop."""
    begin = time.monotonic()
    manifest = check(out)
    H.verify_hashes(out, H.load(out / "01/raw_seal.json"))
    spec, reader = manifest["spec"], H.load(out / "00/reader.json")
    all_rows = H.load(out / "00/rows.json")
    dest = out / "02"
    dest.mkdir()
    interactions, geometry, loops, failures = [], [], [], []
    archives = {}
    for bank, bridge, family in itertools.product(range(2), range(2), range(6)):
        group = bank, bridge, family
        context = {
            "bank": bank,
            "bridge": bridge,
            "current_family": spec["targets"][family],
        }
        rows = [r for r in all_rows if (r["bank"], r["bridge"], r["family"]) == group]
        ambient, vectors, readings, marginal, mixed, q, d = group_data(
            out, rows, reader
        )
        b7, _ = M.support_frame(marginal, spec["span_rtol"])
        b13, _ = M.support_frame(np.column_stack([marginal, mixed]), spec["span_rtol"])
        projected = {
            label: {
                key: ambient.value_jacobian(ambient.x + b @ (b.T @ (x - ambient.x)))[0]
                for key, x in vectors.items()
            }
            for label, b in (("s7", b7), ("s13", b13))
        }
        for axis in range(6):
            _, residual, fraction = M.innovation(mixed[:, axis], b7)
            actual_mixed = M.mixed_contrast(
                readings[axis, 2, 2],
                readings[axis, 0, 2],
                readings[axis, 2, 0],
                readings[axis, 0, 0],
            )
            pm = {
                label: M.mixed_contrast(
                    f[axis, 2, 2], f[axis, 0, 2], f[axis, 2, 0], f[axis, 0, 0]
                )
                for label, f in projected.items()
            }
            for target in range(6):
                interactions.append(
                    {
                        **context,
                        "content_axis": spec["targets"][axis],
                        "target": spec["targets"][target],
                        "unknown_owner_action": float(
                            readings[axis, 2, 1][target] - readings[axis, 0, 1][target]
                        ),
                        "owner_a_action": float(
                            readings[axis, 2, 2][target] - readings[axis, 0, 2][target]
                        ),
                        "owner_b_action": float(
                            readings[axis, 2, 0][target] - readings[axis, 0, 0][target]
                        ),
                        "actual_mixed": float(actual_mixed[target]),
                        "s7_mixed": float(pm["s7"][target]),
                        "s13_mixed": float(pm["s13"][target]),
                        "carrier_mixed_norm": float(np.linalg.norm(mixed[:, axis])),
                        "outside_marginal_span_fraction": fraction,
                        "outside_marginal_span_norm": float(np.linalg.norm(residual)),
                    }
                )
        tag = f"b{bank}_r{bridge}_f{family}"
        archives[tag + "_marginal"] = marginal
        archives[tag + "_mixed"] = mixed
        for label, cols, mode in (
            ("content6", marginal[:, :6], "joint"),
            ("qualified7", marginal, "joint"),
            ("mixed13", np.column_stack([marginal, mixed]), "joint"),
            ("value7", marginal[ambient.size :], "value_only"),
        ):
            b, singular = M.support_frame(cols, spec["span_rtol"])
            rec = {
                **context,
                "chart": label,
                "span_rank": b.shape[1],
                "columns": cols.shape[1],
                "smallest_secant_singular": float(singular[-1]),
                "largest_secant_singular": float(singular[0]),
            }
            try:
                am = (
                    ambient
                    if mode == "joint"
                    else AttentionResponse(ambient.k, ambient.v, q, d, "value_only")
                )
                chart = M.TextSpanResponse(am, b)
                f, j, u, curvature, pairs, condition = chart.brackets(
                    chart.x, spec["gram_rcond"]
                )
                probe = AttentionResponse(
                    am.k, am.v, q.mean(0, keepdims=True), d.mean(0, keepdims=True), mode
                )
                probe_gradient = probe.value_jacobian(probe.x)[1] @ b
                rec.update(
                    status="COMPUTED",
                    response_rank=int(np.linalg.matrix_rank(j)),
                    condition=condition,
                    bracket_norm_rms=float(
                        np.sqrt(np.mean(np.sum(curvature**2, axis=1)))
                    ),
                    bracket_norm_max=float(np.max(np.linalg.norm(curvature, axis=1))),
                    vertical_max=float(np.max(np.abs(j @ curvature.T))),
                    probe_curvature_rms=float(
                        np.sqrt(np.mean((curvature @ probe_gradient.T) ** 2))
                    ),
                )
                archives[tag + "_" + label + "_brackets"] = curvature
                if label == "qualified7":
                    epsilon = (
                        spec["state_radius_fraction"]
                        * np.min(np.linalg.norm(marginal, axis=0))
                        / np.max(np.linalg.norm(u, axis=0))
                    )
                    for pi, (i, k) in enumerate(pairs):
                        loop_conditions = [
                            (s, False) for s in spec["loop_scale_factors"]
                        ]
                        loop_conditions.append((spec["loop_scale_factors"][0], True))
                        for scale, reverse in loop_conditions:
                            dose = epsilon * scale
                            a, c = (k, i) if reverse else (i, k)
                            end = rectangle(
                                chart,
                                chart.x,
                                a,
                                c,
                                dose,
                                spec["rk4_steps_per_leg"],
                                spec["gram_rcond"],
                            )
                            delta = end - chart.x
                            expected = (-1 if reverse else 1) * dose**2 * curvature[pi]
                            norm = float(np.linalg.norm(expected))
                            loops.append(
                                {
                                    **context,
                                    "target_i": spec["targets"][i],
                                    "target_j": spec["targets"][k],
                                    "scale": scale,
                                    "reverse": int(reverse),
                                    "epsilon": dose,
                                    "drift_norm": float(np.linalg.norm(delta)),
                                    "closure_max": float(
                                        np.max(np.abs(chart.value_jacobian(end)[0] - f))
                                    ),
                                    "bracket_relative_error": float(
                                        np.linalg.norm(delta - expected) / norm
                                    )
                                    if norm
                                    else 0.0,
                                    "probe_delta": float(
                                        probe.value_jacobian(chart.lift(end))[0][0]
                                        - probe.value_jacobian(probe.x)[0][0]
                                    ),
                                }
                            )
            except ValueError as error:
                rec.update(
                    status="DECLARED_SUPPORT_FAILED",
                    response_rank=None,
                    condition=None,
                    bracket_norm_rms=None,
                    bracket_norm_max=None,
                    vertical_max=None,
                    probe_curvature_rms=None,
                )
                failures.append({**context, "chart": label, "reason": str(error)})
            geometry.append(rec)
    H.table(dest / "actual_text_interactions.csv", interactions)
    H.table(dest / "response_charts.csv", geometry)
    if loops:
        H.table(dest / "loops.csv", loops)
    with (dest / "secants_and_brackets.npz").open("xb") as stream:
        np.savez_compressed(stream, **archives)
    H.write(dest / "failures.json", failures)
    summary = {
        "packet": "HM104",
        "status": "COMPUTED_FINITE_NOT_GLOBAL",
        "actual_forwards": H.load(out / "01/status.json")["actual_forwards"],
        "model_seconds": H.load(out / "01/status.json")["model_seconds"],
        "analysis_seconds": time.monotonic() - begin,
        "failures": len(failures),
        "contexts": 24,
        "actual_interaction_rows": len(interactions),
        "geometry_rows": len(geometry),
        "loop_rows": len(loops),
        "banks": {},
        "geometry": {},
        "new_fits": 0,
        "natural_continuous_reachability_proven": False,
        "psychological_effect_size": False,
        "original_scope_complete": False,
    }
    for bank in range(2):
        rs = [r for r in interactions if r["bank"] == bank]
        summary["banks"][str(bank)] = {
            name: stat([r[name] for r in rs])
            for name in (
                "actual_mixed",
                "unknown_owner_action",
                "owner_a_action",
                "owner_b_action",
                "outside_marginal_span_fraction",
            )
        }
        for label in ("s7", "s13"):
            summary["banks"][str(bank)][label + "_mixed_error"] = stat(
                [r[label + "_mixed"] - r["actual_mixed"] for r in rs]
            )
    for label in ("content6", "qualified7", "mixed13", "value7"):
        rs = [r for r in geometry if r["chart"] == label and r["status"] == "COMPUTED"]
        summary["geometry"][label] = {
            "computed_contexts": len(rs),
            **{
                name: stat([r[name] for r in rs])
                for name in (
                    "span_rank",
                    "condition",
                    "bracket_norm_rms",
                    "bracket_norm_max",
                    "probe_curvature_rms",
                )
            },
        }
    summary["loops"] = {
        name: stat([r[name] for r in loops])
        for name in (
            "drift_norm",
            "closure_max",
            "bracket_relative_error",
            "probe_delta",
        )
    }
    H.write(out / "summary.json", summary)
    H.write(
        out / "output_seal.json",
        H.seals(out, dest) | {"summary.json": H.sha(out / "summary.json")},
    )
    print("ANALYZED HM104", summary, flush=True)


def verify(out):
    """Check actual corners by alternate arithmetic and chart derivatives numerically."""
    manifest = check(out)
    H.verify_hashes(out, H.load(out / "01/raw_seal.json"))
    H.verify_hashes(out, H.load(out / "output_seal.json"))
    import csv

    with (out / "02/actual_text_interactions.csv").open() as stream:
        recorded = list(csv.DictReader(stream))
    reader, all_rows, spec = (
        H.load(out / "00/reader.json"),
        H.load(out / "00/rows.json"),
        manifest["spec"],
    )
    corner_error = innovation_error = derivative_error = hessian_error = 0.0
    for bank, bridge, family in itertools.product(range(2), range(2), range(6)):
        rows = [
            r
            for r in all_rows
            if (r["bank"], r["bridge"], r["family"]) == (bank, bridge, family)
        ]
        ambient, _, readings, marginal, mixed, _, _ = group_data(out, rows, reader)
        for axis in range(6):
            weights = np.array([1, -1, -1, 1])
            corners = [(axis, 2, 2), (axis, 0, 2), (axis, 2, 0), (axis, 0, 0)]
            alt = weights @ np.stack([readings[key] for key in corners])
            coefficients = np.linalg.lstsq(
                marginal, mixed[:, axis], rcond=spec["span_rtol"]
            )[0]
            norm = np.linalg.norm(mixed[:, axis])
            fraction = (
                np.linalg.norm(mixed[:, axis] - marginal @ coefficients) / norm
                if norm
                else 0.0
            )
            rs = [
                r
                for r in recorded
                if int(r["bank"]) == bank
                and int(r["bridge"]) == bridge
                and r["current_family"] == spec["targets"][family]
                and r["content_axis"] == spec["targets"][axis]
            ]
            for target, row in enumerate(rs):
                if row["target"] != spec["targets"][target]:
                    raise ValueError("Target ordering changed")
                corner_error = max(
                    corner_error, abs(float(row["actual_mixed"]) - alt[target])
                )
                innovation_error = max(
                    innovation_error,
                    abs(float(row["outside_marginal_span_fraction"]) - fraction),
                )
        b, _ = M.support_frame(marginal, spec["span_rtol"])
        chart = M.TextSpanResponse(ambient, b)
        a = np.arange(1, b.shape[1] + 1, dtype=float)
        a /= np.linalg.norm(a)
        h = spec["finite_difference_step"]
        fplus, jplus = chart.value_jacobian(chart.x + h * a)
        fminus, jminus = chart.value_jacobian(chart.x - h * a)
        _, j = chart.value_jacobian(chart.x)
        derivative_error = max(
            derivative_error, float(np.max(np.abs((fplus - fminus) / (2 * h) - j @ a)))
        )
        hessian_error = max(
            hessian_error,
            float(
                np.max(
                    np.abs(
                        (jplus - jminus) / (2 * h)
                        - chart.directional_jacobian(chart.x, a)
                    )
                )
            ),
        )
    errors = {
        "actual_corner_replay_max": corner_error,
        "least_squares_innovation_max": innovation_error,
        "chart_central_gradient_max": derivative_error,
        "chart_central_hessian_max": hessian_error,
    }
    if max(errors.values()) > spec["numeric_atol"]:
        H.write(out / "verification_failure.json", errors)
        raise ValueError("Finite verification failed; retain original artifacts")
    H.write(
        out / "verification.json",
        {
            "status": "VERIFIED_FINITE_HM104_NOT_GLOBAL",
            **errors,
            "actual_captures_checked": len(all_rows),
            "same_agent_alternate_arithmetic": True,
            "new_model_calls": 0,
        },
    )
    print("VERIFIED HM104", errors, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "collect", "analyze", "verify"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = H.load(SPEC)
    target = args.output or ROOT / settings["output"]
    if args.stage == "prepare":
        prepare(target, settings)
    else:
        {"collect": collect, "analyze": analyze, "verify": verify}[args.stage](target)
