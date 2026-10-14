#!/usr/bin/env python3
"""One source-to-vector observer construction, not a prediction/precision search."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_vector_lift as V
from scripts import run_grain_configuration_kernel_hm144 as K
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_vector_lift_hm181.json"
DEFAULT = ROOT / "results/grain_math_response_v1/actual_vector_target_action_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_VECTOR_LIFT_PLAN_HM181.md",
    "grain_research/construction/math_vector_lift.py",
    "scripts/run_grain_vector_lift_hm181.py",
    "scripts/score_grain_vector_lift_cold.py",
    "scripts/verify_grain_vector_lift_hm181.py",
    "tests/test_grain_vector_lift.py",
)


def prepare(out):
    """Bind old science,actual vector identities and finite scope before outcomes."""
    spec = R.read(ROOT / CONFIG)
    parent, event = ROOT / spec["parent"], ROOT / spec["event_parent"]
    code, inputs = {}, dict.fromkeys(K.INPUTS.values())
    for par, stages in (
        (
            parent,
            (
                "compiled_seal.json",
                "scored_seal.json",
                "analysis_seal.json",
                "trade_verification_seal.json",
            ),
        ),
        (event, ("compiled_seal.json", "cold_seal.json", "analysis_seal.json")),
    ):
        m = R.check(par, stages)
        code.update(m["code"])
        for stage in ("manifest.json", "prepared_seal.json", *stages):
            files = (
                (stage,) if stage == "manifest.json" else (stage, *R.read(par / stage))
            )
            for name in files:
                inputs[str((par / name).relative_to(ROOT))] = None
    inputs[str((event / "source_law.json").relative_to(ROOT))] = None
    a, e = R.prepared(parent), R.prepared(event)
    inv = np.argsort(e["original_units"], axis=1)
    coords = np.take_along_axis(e["coordinates"], inv[:, :, None], axis=1)
    cond = np.take_along_axis(e["conditions"], inv, axis=1)
    threshold = R.read(event / "source_law.json")["thresholds"]
    cells = 4 * cond + R.F.states(coords.reshape(-1, 15), threshold).reshape(624, 36)
    expected_cells = np.stack(
        (cells[a["source"], :12], cells[a["source"], 12:24]), axis=1
    )
    if not np.array_equal(expected_cells, a["source_cells"]):
        raise ValueError("Actual source event-to-cell identities changed")
    embeddings, directions, _, _, _, _ = K.load_numeric()
    identity = float(np.max(np.abs(embeddings @ directions - coords)))
    if identity > spec["comparison_tolerance"]:
        raise ValueError("Full E5-to-GRAIN observation identity changed")
    source, panel = a["source"], a["application_indices"]
    if len(source) != 386 or len(panel) != 24 or set(source) & set(panel):
        raise ValueError("Historical source/panel author scope changed")
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": spec,
            "code": code | {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {p: R.digest(ROOT / p) for p in inputs},
            "original_scope_complete": False,
            "historical_cohorts_not_lockbox": True,
        },
    )
    R.archive(
        out / "observations.npz",
        source=source,
        panel=panel,
        source_coordinates=coords[source, :24].reshape(386, 2, 12, 15),
        panel_coordinates=coords[panel, :24].reshape(24, 2, 12, 15),
        cells=a["source_cells"],
        targets=a["source_targets"],
    )
    count = 0
    with (out / "cold_input.jsonl").open("x") as sink:
        for line in (parent / "cold_input.jsonl").read_text().splitlines():
            old = json.loads(line)
            arm, u, v, subset = map(int, old["author_key"].split("/"))
            if arm != 1:
                continue
            record = {
                "frame_key": f"{u}/{v}/{subset}",
                "view": v,
                "events": [
                    {
                        "event_id": t["event_id"],
                        "slot": t["slot"],
                        "coordinates": coords[panel[u], 12 * v + t["slot"]].tolist(),
                    }
                    for t in old["events"]
                ],
            }
            sink.write(json.dumps(record) + "\n")
            count += 1
    if count != spec["cold_records"]:
        raise ValueError("Incomplete cold finite scope")
    R.write(
        out / "quality.json",
        {
            "source_authors": 386,
            "panel_authors": 24,
            "views": 2,
            "source_triples": 49408,
            "cold_records": count,
            "E5_GRAIN_identity_error": identity,
            "new_neural_API_private_raw_or_application_label_reads": 0,
            "source_labels_reused": "HM180 authorized binary Big5 SOURCE only",
            "order_is_original_slot_not_clinical_time": True,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "cold_input.jsonl", "quality.json"),
    )
    print(json.dumps(R.read(out / "quality.json")))


def compile_observer(out):
    """Lift frozen weights once and compile actual-vector target witness functions."""
    spec = R.check(out)["config"]
    z, began = R.prepared(out), time.perf_counter()
    parent = ROOT / spec["parent"]
    b = R.read(parent / "source_bundle.json")
    full = np.asarray(b["profile_tables"], float)[:, 1]
    with np.load(parent / "source_trade_components.npz", allow_pickle=False) as a:
        changes = a["changes"]
    h, _, _, _, _, _ = K.load_numeric()
    h = h[z["source"], :24].reshape(386, 2, 12, 1024)
    results = [
        V.source_moments(
            z["source_coordinates"][:, v],
            h[:, v],
            z["cells"][:, v],
            z["targets"],
            b["profiles"],
            full[v],
            changes[v],
        )
        for v in range(2)
    ]
    arrays = {k: np.stack([r[k] for r in results]) for k in results[0]}
    R.archive(out / "source_footprints.npz", **arrays)
    moments, e5, coarse = arrays["moments"], arrays["e5"], arrays["coarse"]
    witnesses = moments[:, np.arange(5), np.arange(5) + 1]
    bundle = {
        "version": V.VERSION,
        "code_hash": V.code_hash(),
        "targets": list(V.TARGETS),
        "dimensions": 15,
        "witnesses": witnesses.tolist(),
        "interpretation": "source_signed_tensor_reading;not_probability_or_text_intervention",
    }
    V.validate_bundle(bundle)
    R.write(out / "vector_bundle.json", bundle)
    rows, sector = [], V.sectors()
    for v, q, r, order in np.ndindex(2, 5, 6, 4):
        w = moments[v, q, r, sector == order]
        rows.append(
            {
                "view": v,
                "action": V.TARGETS[q],
                "probe": ("unconditional", *V.TARGETS)[r],
                "order": order,
                "coordinates": len(w),
                "norm": float(np.linalg.norm(w)),
                "max_absolute": float(np.max(np.abs(w))),
                "coarse_preservation_expected": r != q + 1,
            }
        )
    R.table(out / "fine_tensor_footprints.csv", rows)
    erows = []
    for v, q, r, role in np.ndindex(2, 5, 6, 3):
        w = e5[v, q, r, role]
        erows.append(
            {
                "view": v,
                "action": V.TARGETS[q],
                "probe": ("unconditional", *V.TARGETS)[r],
                "role": role,
                "norm": float(np.linalg.norm(w)),
                "max_absolute": float(np.max(np.abs(w))),
            }
        )
    R.table(out / "E5_role_footprints.csv", erows)
    grow = []
    for q, p, order in np.ndindex(5, 5, 4):
        a, c = witnesses[0, q, sector == order], witnesses[1, p, sector == order]
        an, cn = np.linalg.norm(a), np.linalg.norm(c)
        grow.append(
            {
                "source_view0_target": V.TARGETS[q],
                "source_view1_target": V.TARGETS[p],
                "order": order,
                "signed_inner": float(a @ c),
                "cosine": float(a @ c / (an * cn))
                if order and an > 0 and cn > 0
                else None,
            }
        )
    R.table(out / "source_view_cross_gram.csv", grow)
    error_f, error_h = 0.0, 0.0
    for v in range(2):
        f = np.bincount(arrays["flat"][v].ravel(), minlength=32 * 512).reshape(
            32, 8, 8, 8
        ) / (386 * 64)
        hh = np.bincount(
            arrays["flat"][v].ravel(),
            weights=arrays["lifted"][v].ravel(),
            minlength=32 * 512,
        ).reshape(32, 8, 8, 8)
        error_f = max(error_f, float(np.max(np.abs(f - full[v]))))
        error_h = max(
            error_h, float(np.max(np.abs(hh - np.asarray(b["profile_tables"])[v, 0])))
        )
    off = np.ones((5, 6), bool)
    off[np.arange(5), np.arange(5) + 1] = False
    receipt = {
        "method": "canonical_conditional_source_support_lift_and_signed_tensor_reader",
        "seconds": time.perf_counter() - began,
        "source_pushforward_F_error": error_f,
        "source_pushforward_H_error": error_h,
        "lift_ratio_min": float(arrays["lifted"].min() * 386 * 64),
        "lift_ratio_max": float(arrays["lifted"].max() * 386 * 64),
        "coarse_other_target_max": float(np.max(np.abs(coarse[:, off]))),
        "fine_other_or_unconditional_max": float(np.max(np.abs(moments[:, off, 1:]))),
        "fine_other_or_unconditional_E5_max": float(np.max(np.abs(e5[:, off]))),
        "label_blind_coarse_density_max": float(
            max(
                np.max(
                    np.abs(
                        np.divide(
                            coarse[v, :, 0],
                            full[v].sum(0).ravel()[None, :],
                            out=np.zeros_like(coarse[v, :, 0]),
                            where=full[v].sum(0).ravel()[None, :] > 0,
                        )
                    )
                )
                for v in range(2)
            )
        ),
        "all_source_fine_probe_rows": len(rows),
        "all_E5_probe_rows": len(erows),
        "no_new_model_no_optimization": True,
        "actual_text_intervention_claim": False,
    }
    if (
        max(error_f, error_h, receipt["coarse_other_target_max"])
        > spec["comparison_tolerance"]
    ):
        raise ValueError("Source-supported lift violated its fixed coarse contract")
    R.write(out / "construction_receipt.json", receipt)
    R.seal(
        out,
        "compiled_seal.json",
        (
            "source_footprints.npz",
            "vector_bundle.json",
            "fine_tensor_footprints.csv",
            "E5_role_footprints.csv",
            "source_view_cross_gram.csv",
            "construction_receipt.json",
        ),
    )
    print(json.dumps(receipt))


def score(out):
    """Execute all cold inputs separately before judging fine-resolution meaning."""
    R.check(out, ("compiled_seal.json",))
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_vector_lift_cold.py"),
            "--bundle",
            str(out / "vector_bundle.json"),
            "--input",
            str(out / "cold_input.jsonl"),
            "--output",
            str(out / "cold_output.jsonl"),
        ],
        check=True,
    )
    records = [
        json.loads(s) for s in (out / "cold_output.jsonl").read_text().splitlines()
    ]
    if len(records) != 240 or len({r["frame_key"] for r in records}) != 240:
        raise ValueError("Cold addressed scope changed")
    rows = []
    for record in records:
        for source_view, q in itertools.product(range(2), range(5)):
            rows.append(
                {
                    "frame_key": record["frame_key"],
                    "text_view": record["view"],
                    "source_view": source_view,
                    "target": V.TARGETS[q],
                    "triples": record["triples"],
                    "retained_reading": record["retained_readings"][source_view][q],
                    "conditional_reading": record["conditional_readings"][source_view][
                        q
                    ]
                    if record["triples"]
                    else None,
                }
            )
    R.table(out / "cold_vector_readings.csv", rows)
    c = R.read(out / "construction_receipt.json")
    summary = c | {
        "cold_records": len(records),
        "cold_values": len(rows),
        "no_triple_inputs": sum(r["triples"] == 0 for r in records),
        "original_scope_complete": False,
        "mechanism": "FINE_OBSERVER_BREAKS_COARSE_ISOLATION"
        if c["fine_other_or_unconditional_max"] > 1e-12
        else "NO_FINE_FOOTPRINT_IN_REGISTERED_PROBES",
        "reader_is_not_probability_or_causal_personality_change": True,
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "scored_seal.json",
        ("cold_output.jsonl", "cold_vector_readings.csv", "summary.json"),
    )
    print(json.dumps(summary))


def main():
    """Execute one bounded stage; preserve any failure without automatic retries."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "compile", "score"))
    p.add_argument("--out", type=Path, default=DEFAULT)
    a = p.parse_args()
    try:
        {"prepare": prepare, "compile": compile_observer, "score": score}[a.stage](
            a.out
        )
    except Exception as error:
        if a.out.exists() and not (a.out / f"{a.stage}_failure.json").exists():
            R.write(
                a.out / f"{a.stage}_failure.json", {"error": str(error), "retry": False}
            )
        raise


if __name__ == "__main__":
    main()
