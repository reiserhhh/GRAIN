#!/usr/bin/env python3
"""Source-frozen joint evidence on identical current-token carriers (HM105)."""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_author_evidence_merge as A
from grain_research.construction import math_history_routing as R
from grain_research.construction import math_qualified_transport as M
from grain_research.construction.math_response_connection import AttentionResponse
from grain_research.construction.math_text_response import support_frame
from grain_research.vector_mechanisms.navigation import MODEL, write_json
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts import run_grain_text_response_hm104 as T
from scripts.run_grain_history_routing_hm101 import check as source_check
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_qualified_transport_hm105.json"
CODE = (
    "configs/grain_qualified_transport_hm105.json",
    "docs/GRAIN_QUALIFIED_TRANSPORT_PLAN_HM105.md",
    "grain_research/construction/math_qualified_transport.py",
    "grain_research/construction/math_text_response.py",
    "grain_research/construction/math_response_connection.py",
    "grain_research/construction/math_author_evidence_merge.py",
    "grain_research/construction/math_history_routing.py",
    "grain_research/vector_mechanisms/prefix.py",
    "scripts/run_grain_qualified_transport_hm105.py",
    "scripts/run_grain_text_response_hm104.py",
    "scripts/run_grain_author_evidence_merge_hm102.py",
    "scripts/run_grain_math_span_binding.py",
    "scripts/run_grain_upstream_routes.py",
    "tests/test_grain_math_qualified_transport.py",
)


def check(out):
    """Authenticate fixed code, public material and reader before every phase."""
    manifest = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, manifest["code"] | manifest["source"])
    H.verify_hashes(out, manifest["inputs"])
    return manifest


def prepare(out, spec):
    """Freeze material and prove cross-bank current and name-bag alignment."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Keep all previous and partial roots")
    source = ROOT / spec["source"]
    source_check(source)
    H.verify_hashes(source, H.load(source / "01/raw_seal.json"))
    if (
        H.load(source / "verification.json")["status"]
        != "VERIFIED_FINITE_HM101_NOT_GLOBAL"
    ):
        raise ValueError("Verified reader source required")
    bundle = H.load(source / "00/bundle.json")
    if bundle["targets"] != spec["targets"]:
        raise ValueError("Target order must remain fixed")
    old = H.load(ROOT / spec["source_material"])
    known = H.load(ROOT / spec["known_application_material"])
    banks = [
        {t: old["behaviours"][t][0] for t in spec["targets"]},
        known["behaviours"],
        spec["new_behaviours"],
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

    def tokens(text):
        """Use fixed segmented tokenization, never target answers."""
        return tokenizer.encode(text, add_special_tokens=False)

    prefix = tokens(rendered.split("__GRAIN_BODY__")[0])
    histories = {}
    for bank, (axis, level, owner) in itertools.product(range(3), list(T.design())):
        phrases = [
            banks[bank][t][level if i == axis else 1]
            for i, t in enumerate(spec["targets"])
        ]
        text = "\n".join(
            f"Earlier entry E{i + 1}: the person {p}." for i, p in enumerate(phrases)
        )
        text += "\n" + spec["owners"][owner]
        histories[bank, axis, level, owner] = {"text": text, "tokens": tokens(text)}
    bag_checks = 0
    for (bank, axis, level, owner), history in histories.items():
        if owner == 0:
            other = histories[bank, axis, level, 2]
            if Counter(history["tokens"]) != Counter(other["tokens"]):
                raise ValueError("Owner contrast changes the token bag")
            bag_checks += 1
    width = max(len(h["tokens"]) for h in histories.values())
    rows, identities = [], {}
    for bank, bridge, carrier in itertools.product(range(3), range(2), range(2)):
        bt = tokens("\n" + spec["bridges"][bridge] + "\nCurrent entry: Person A")
        ct = tokens(spec["currents"][carrier])
        current = list(
            range(
                len(prefix) + width + len(bt), len(prefix) + width + len(bt) + len(ct)
            )
        )
        identity = ct, current
        if (bridge, carrier) in identities and identities[bridge, carrier] != identity:
            raise ValueError("Current IDs/positions change between banks")
        identities[bridge, carrier] = identity
        for axis, level, owner in T.design():
            h = histories[bank, axis, level, owner]
            padding = list(range(len(prefix) + len(h["tokens"]), len(prefix) + width))
            ids = (
                prefix
                + h["tokens"]
                + [tokenizer.pad_token_id] * (width - len(h["tokens"]))
                + bt
                + ct
            )
            history = list(range(len(prefix), len(prefix) + width))
            excluded = set(history) | set(current)
            relay = [i for i in range(len(ids)) if i not in excluded]
            rows.append(
                {
                    "unit": len(rows),
                    "bank": bank,
                    "bridge": bridge,
                    "carrier": carrier,
                    "axis": axis,
                    "level": level,
                    "owner": owner,
                    "history_text": h["text"],
                    "current_text": spec["currents"][carrier],
                    "input_ids": ids,
                    "history": history,
                    "relay": relay,
                    "current": current,
                    "padding": padding,
                }
            )
    if len(rows) != spec["max_forwards"] or len(histories) != 117 or bag_checks != 39:
        raise ValueError("Declared finite battery incomplete")
    (out / "00").mkdir(parents=True)
    (out / "01/units").mkdir(parents=True)
    H.write(out / "00/rows.json", rows)
    H.write(
        out / "00/reader.json",
        {
            "targets": bundle["targets"],
            "directions": bundle["directions"],
            "positions": {
                str(e): A.registered_queries(bundle, e).tolist()
                for e in sorted({max(r["current"]) for r in rows})
            },
        },
    )
    sources = [
        source / "00/bundle.json",
        source / "verification.json",
        MODEL / "config.json",
        MODEL / "tokenizer.json",
    ]
    sources += [
        ROOT / spec[name]
        for name in ("source_material", "known_application_material", "system_material")
    ]
    H.write(
        out / "00/manifest.json",
        {
            "packet": "HM105",
            "spec": spec,
            "history_width": width,
            "actual_forwards_planned": len(rows),
            "name_token_bag_checks": bag_checks,
            "common_carrier_checks": len(identities),
            "new_fit": False,
            "code": {name: H.sha(ROOT / name) for name in CODE},
            "source": {str(p.relative_to(ROOT)): H.sha(p) for p in sources},
            "inputs": {
                f"00/{n}.json": H.sha(out / f"00/{n}.json") for n in ("rows", "reader")
            },
        },
    )
    print(
        "PREPARED HM105",
        {"forwards": len(rows), "width": width, "name_bags": bag_checks},
        flush=True,
    )


def collect(out):
    """Run only the finite local prefix battery and preserve per-unit recovery."""
    manifest = check(out)
    if (out / "01/raw_seal.json").exists():
        raise FileExistsError("Do not rerun completed capture")
    rows = H.load(out / "00/rows.json")
    path = out / "01/status.json"
    status = (
        H.load(path)
        if path.exists()
        else {"completed": 0, "model_seconds": 0.0, "state": "RUNNING"}
    )
    observer = load_observer()
    with collector_guard(out):
        for row in rows:
            stem = out / f"01/units/unit_{row['unit']:04d}"
            if stem.with_suffix(".json").exists():
                T.raw_unit(out, row)
                continue
            if status["model_seconds"] >= manifest["spec"]["max_model_seconds"]:
                raise TimeoutError("Finite model budget reached; retain checkpoint")
            start = time.monotonic()
            raw, trace = R.capture_routed_prefix(observer, row, 1, 1)
            seconds = time.monotonic() - start
            with stem.with_suffix(".npz").open("xb") as stream:
                np.savez_compressed(stream, **raw)
            H.write(
                stem.with_suffix(".json"),
                {
                    "unit": row["unit"],
                    "trace": trace,
                    "seconds": seconds,
                    "sha256": H.sha(stem.with_suffix(".npz")),
                },
            )
            status.update(
                completed=row["unit"] + 1,
                model_seconds=status["model_seconds"] + seconds,
            )
            write_json(path, status)
            if status["completed"] % 78 == 0:
                print("HM105", status, flush=True)
        status.update(state="COLLECTED", actual_forwards=len(rows))
        write_json(path, status)
        H.write(out / "01/raw_seal.json", H.seals(out, out / "01/units"))
    print("COLLECTED HM105", status, flush=True)


def group(out, rows, reader):
    """Return aligned actual endpoints, readings and whole marginal/mixed spans."""
    identity, vectors, raw = None, {}, {}
    for row in rows:
        current = [row["input_ids"][p] for p in row["current"]], row["current"]
        if identity is not None and current != identity:
            raise ValueError("Carrier coordinates are not common")
        identity = current
        key = row["axis"], row["level"], row["owner"]
        data = T.raw_unit(out, row)
        raw[key] = data
        vectors[key] = np.concatenate((data["k"].ravel(), data["v"].ravel()))
    base = raw[-1, 1, 1]
    q = np.asarray(reader["positions"][str(max(rows[0]["current"]))], float)
    observer = AttentionResponse(
        base["k"], base["v"], q, np.asarray(reader["directions"])
    )
    readings = {key: observer.value_jacobian(x)[0] for key, x in vectors.items()}
    marginals = [vectors[i, 2, 1] - vectors[i, 0, 1] for i in range(6)]
    marginals.append(vectors[-1, 1, 2] - vectors[-1, 1, 0])
    mixed = np.stack([M.SIGNS @ corners(vectors, i) for i in range(6)], axis=1)
    return observer, vectors, readings, np.stack(marginals, axis=1), mixed


def corners(values, axis):
    """Gather a complete ordered corner group; absent observations fail."""
    return np.stack(
        [
            values[axis, level, owner]
            for level, owner in ((2, 2), (0, 2), (2, 0), (0, 0))
        ]
    )


def source_frames(out):
    """Seal all source dictionaries before reading any application captures."""
    manifest = check(out)
    H.verify_hashes(out, H.load(out / "01/raw_seal.json"))
    rows, reader = H.load(out / "00/rows.json"), H.load(out / "00/reader.json")
    dest = out / "02_source"
    dest.mkdir()
    records = []
    for bridge, carrier in itertools.product(range(2), range(2)):
        source_rows = [
            r
            for r in rows
            if (r["bank"], r["bridge"], r["carrier"]) == (0, bridge, carrier)
        ]
        _, _, _, marginal, mixed = group(out, source_rows, reader)
        s7, v7 = support_frame(marginal, manifest["spec"]["span_rtol"])
        s13, v13 = support_frame(
            np.column_stack((marginal, mixed)), manifest["spec"]["span_rtol"]
        )
        with (dest / f"b{bridge}_c{carrier}.npz").open("xb") as stream:
            np.savez_compressed(stream, s7=s7, s13=s13, marginal=marginal, mixed=mixed)
        records.append(
            {
                "bridge": bridge,
                "carrier": carrier,
                "rank7": s7.shape[1],
                "rank13": s13.shape[1],
                "singular7": v7.tolist(),
                "singular13": v13.tolist(),
            }
        )
    H.write(
        dest / "manifest.json",
        {"source_bank": 0, "application_read": False, "records": records},
    )
    H.write(out / "source_seal.json", H.seals(out, dest))
    print("SOURCE FROZEN HM105", records, flush=True)


def analyze(out):
    """Test frozen geometry and its target consequences, without outcome fitting."""
    start = time.monotonic()
    manifest = check(out)
    H.verify_hashes(out, H.load(out / "01/raw_seal.json"))
    H.verify_hashes(out, H.load(out / "source_seal.json"))
    rows, reader = H.load(out / "00/rows.json"), H.load(out / "00/reader.json")
    dest = out / "03"
    dest.mkdir()
    interactions, geometry, angles, profiles = [], [], [], []
    marginal_error = 0.0
    for bridge, carrier in itertools.product(range(2), range(2)):
        with np.load(
            out / f"02_source/b{bridge}_c{carrier}.npz", allow_pickle=False
        ) as z:
            frame = {k: np.array(z[k]) for k in z.files}
        for bank in range(3):
            selected = [
                r
                for r in rows
                if (r["bank"], r["bridge"], r["carrier"]) == (bank, bridge, carrier)
            ]
            observer, vectors, readings, _, mixed = group(out, selected, reader)
            common = {"bank": bank, "bridge": bridge, "carrier": carrier}
            for i, cosine in enumerate(
                M.principal_cosines(
                    frame["mixed"], mixed, manifest["spec"]["span_rtol"]
                )
            ):
                angles.append({**common, "principal_index": i, "cosine": float(cosine)})
            for axis in range(6):
                x, actual = corners(vectors, axis), corners(readings, axis)
                interaction = M.SIGNS @ actual
                authored = M.paired_profile_contrast(actual)
                g = {
                    **common,
                    "axis": axis,
                    "axis_name": reader["targets"][axis],
                    "mixed_norm": float(np.linalg.norm(mixed[:, axis])),
                    "source_axis_cosine": M.vector_cosine(
                        frame["mixed"][:, axis], mixed[:, axis]
                    ),
                }
                for label in ("s7", "s13"):
                    edited, _, residual = M.project_joint_corners(x, frame[label])
                    marginal_error = max(
                        marginal_error,
                        float(
                            np.max(
                                np.abs(M.marginal_means(x) - M.marginal_means(edited))
                            )
                        ),
                    )
                    diagnostic = np.stack(
                        [observer.value_jacobian(v)[0] for v in edited]
                    )
                    projected = M.SIGNS @ diagnostic
                    g[f"outside_{label}"] = float(
                        np.linalg.norm(residual) / np.linalg.norm(mixed[:, axis])
                    )
                    for target in range(6):
                        interactions.append(
                            {
                                **common,
                                "axis": axis,
                                "target": reader["targets"][target],
                                "source_span": label,
                                "actual": float(interaction[target]),
                                "diagnostic": float(projected[target]),
                                "removed_response": float(
                                    interaction[target] - projected[target]
                                ),
                            }
                        )
                geometry.append(g)
                for target in range(6):
                    profiles.append(
                        {
                            **common,
                            "axis": axis,
                            "target": reader["targets"][target],
                            "actual_profile_contrast": float(authored[target]),
                            "quarter_interaction": float(interaction[target] / 4),
                        }
                    )
    H.table(dest / "interactions.csv", interactions)
    H.table(dest / "geometry.csv", geometry)
    H.table(dest / "principal_angles.csv", angles)
    H.table(dest / "author_profiles.csv", profiles)
    summary = {
        "packet": "HM105",
        "status": "COMPUTED_FINITE_NOT_GLOBAL",
        "actual_forwards": len(rows),
        "model_seconds": H.load(out / "01/status.json")["model_seconds"],
        "analysis_seconds": time.monotonic() - start,
        "marginal_preservation_max": marginal_error,
        "new_predictor_fits": 0,
        "new_model_calls_in_analysis": 0,
        "banks": {},
        "original_scope_complete": False,
    }
    for bank in range(3):
        ir = [r for r in interactions if r["bank"] == bank]
        gr = [r for r in geometry if r["bank"] == bank]
        summary["banks"][str(bank)] = {
            "actual_interaction": T.stat(
                [r["actual"] for r in ir if r["source_span"] == "s7"]
            ),
            "source_axis_cosine": T.stat([r["source_axis_cosine"] for r in gr]),
            "principal_cosines": T.stat(
                [r["cosine"] for r in angles if r["bank"] == bank]
            ),
            "outside_s7": T.stat([r["outside_s7"] for r in gr]),
            "outside_s13": T.stat([r["outside_s13"] for r in gr]),
            "removed_response_s7": T.stat(
                [r["removed_response"] for r in ir if r["source_span"] == "s7"]
            ),
            "removed_response_s13": T.stat(
                [r["removed_response"] for r in ir if r["source_span"] == "s13"]
            ),
            "profile_contrast": T.stat(
                [r["actual_profile_contrast"] for r in profiles if r["bank"] == bank]
            ),
        }
    H.write(out / "summary.json", summary)
    H.write(
        out / "output_seal.json",
        {**H.seals(out, dest), "summary.json": H.sha(out / "summary.json")},
    )
    print("ANALYZED HM105", summary, flush=True)


def verify(out):
    """Replay actual readings and mixed projection by separate arithmetic."""
    import csv

    manifest = check(out)
    for seal in ("01/raw_seal.json", "source_seal.json", "output_seal.json"):
        H.verify_hashes(out, H.load(out / seal))
    rows, reader = H.load(out / "00/rows.json"), H.load(out / "00/reader.json")
    maximum = {
        "actual_reading": 0.0,
        "least_squares_residual": 0.0,
        "marginal_preservation": 0.0,
        "projection_reading": 0.0,
        "profile_identity": 0.0,
    }
    with (out / "03/interactions.csv").open() as stream:
        table = {
            (
                int(r["bank"]),
                int(r["bridge"]),
                int(r["carrier"]),
                int(r["axis"]),
                r["target"],
                r["source_span"],
            ): r
            for r in csv.DictReader(stream)
        }
    for bridge, carrier in itertools.product(range(2), range(2)):
        with np.load(
            out / f"02_source/b{bridge}_c{carrier}.npz", allow_pickle=False
        ) as z:
            frame = {k: np.array(z[k]) for k in z.files}
        for bank in range(3):
            selected = [
                r
                for r in rows
                if (r["bank"], r["bridge"], r["carrier"]) == (bank, bridge, carrier)
            ]
            observer, vectors, readings, _, _ = group(out, selected, reader)
            for axis in range(6):
                x = corners(vectors, axis)
                actual = []
                for v in x:
                    k, value = observer.unpack(v)
                    actual.append(A.observe(k, value, observer.q, observer.d)[0].sum(1))
                actual = np.stack(actual)
                maximum["actual_reading"] = max(
                    maximum["actual_reading"],
                    float(np.max(np.abs(actual - corners(readings, axis)))),
                )
                signs = np.array([1.0, -1.0, -1.0, 1.0])
                w = x[0] - x[1] - x[2] + x[3]
                contrast = (actual[0] - actual[1]) - (actual[2] - actual[3])
                profile = ((1 + 0.5 * signs) / 4) @ actual - (
                    (1 - 0.5 * signs) / 4
                ) @ actual
                maximum["profile_identity"] = max(
                    maximum["profile_identity"],
                    float(np.max(np.abs(profile - contrast / 4))),
                )
                for label in ("s7", "s13"):
                    b = frame[label]
                    coeff = np.linalg.lstsq(b, w, rcond=None)[0]
                    residual = w - b @ coeff
                    diagnostic = x - signs[:, None] * residual[None] / 4
                    main, _, main_residual = M.project_joint_corners(x, b)
                    maximum["least_squares_residual"] = max(
                        maximum["least_squares_residual"],
                        float(np.max(np.abs(residual - main_residual))),
                    )
                    maximum["marginal_preservation"] = max(
                        maximum["marginal_preservation"],
                        float(
                            np.max(np.abs(M.marginal_means(x) - M.marginal_means(main)))
                        ),
                    )
                    f = []
                    for v in diagnostic:
                        k, value = observer.unpack(v)
                        f.append(A.observe(k, value, observer.q, observer.d)[0].sum(1))
                    f = np.stack(f)
                    projected = (f[0] - f[1]) - (f[2] - f[3])
                    for target, name in enumerate(reader["targets"]):
                        r = table[bank, bridge, carrier, axis, name, label]
                        maximum["projection_reading"] = max(
                            maximum["projection_reading"],
                            abs(float(r["actual"]) - contrast[target]),
                            abs(float(r["diagnostic"]) - projected[target]),
                        )
    if max(maximum.values()) > manifest["spec"]["arithmetic_atol"]:
        raise ValueError("Finite alternate-arithmetic verifier failed: " + str(maximum))
    result = {
        "status": "VERIFIED_FINITE_HM105_NOT_GLOBAL",
        "max_errors": maximum,
        "actual_captures_checked": len(rows),
        "source_frames_frozen_before_application_read": True,
        "same_agent_alternate_arithmetic": True,
        "new_model_calls": 0,
    }
    H.write(out / "verification.json", result)
    print("VERIFIED HM105", result, flush=True)


def main():
    """Execute one bounded phase; never overwrite a completed canonical stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=("prepare", "collect", "source", "analyze", "verify")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    spec = H.load(SPEC)
    out = args.output or ROOT / spec["output"]
    if args.phase == "prepare":
        prepare(out, spec)
    else:
        {
            "collect": collect,
            "source": source_frames,
            "analyze": analyze,
            "verify": verify,
        }[args.phase](out)


if __name__ == "__main__":
    main()
