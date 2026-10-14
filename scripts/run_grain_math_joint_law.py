#!/usr/bin/env python3
"""HM10A: real text-vector joint laws, independent lexical-pair source support."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from run_grain_math_binding import load_model, read_bank
from run_grain_math_local_roles import vectors
from run_grain_math_response import MODELS, ROOT, encode, sha, write_csv, write_json
from run_grain_math_state_exposure import check_source, coverage_audit, parse

from grain_research.construction import math_binding as B
from grain_research.construction import math_joint_law as J
from grain_research.construction import math_local as L
from grain_research.construction import math_state_exposure as S

SPEC = ROOT / "configs/grain_math_joint_law_v1.json"
DEFAULT = ROOT / "results/grain_math_response_v1/joint_law_r1"


def material_bank(singles, spec):
    """Cross conditions and report incidence; do not use eventual model scores."""
    slots = {}
    for target in S.TARGETS:
        source = [
            r
            for r in singles
            if r["target"] == target
            and r["role"] == "source"
            and r["bits"][1:] == [1, 1, 1]
        ]
        domain = []
        for ai in range(4):
            positive = min(
                (r for r in source if r["action_index"] == ai),
                key=lambda r: r["object_index"],
            )
            negative = next(
                r
                for r in singles
                if r["target"] == target
                and r["role"] == "source"
                and r["action_index"] == ai + 4
                and r["object_index"] == positive["object_index"]
                and r["bits"][1:] == [1, 1, 1]
            )
            domain.append((negative, positive))
        slots[target] = domain
    rows = []
    for c in range(4):
        for v in range(4):
            family = 4 * c + v
            role = (
                "source"
                if (c + v) % 4 < 2
                else "validation"
                if (c + v) % 4 == 2
                else "application"
            )
            for frame, states in enumerate(spec["frames"]):
                for actor in spec["trigger_actors"]:
                    for mask in range(8):
                        cb, vb, incidence = [(mask >> j) & 1 for j in range(3)]
                        selected = [
                            slots[S.TARGETS[0]][c][cb],
                            slots[S.TARGETS[1]][v][vb],
                            slots[S.TARGETS[0]][c][0],
                            slots[S.TARGETS[1]][v][0],
                        ]
                        groups = [[0, 1], [2, 3]] if incidence else [[0, 3], [2, 1]]
                        text, reports = "", []
                        for ri, group in enumerate(groups):
                            start = len(text)
                            text += "I feel " + states[ri]
                            report = {
                                "root_span": [start + 2, start + 6],
                                "experiencer": "self",
                                "state_polarity": 1,
                                "triggers": [],
                            }
                            for ti, si in enumerate(group):
                                slot = selected[si]
                                text += " when " if ti == 0 else " and when "
                                text += "I " if actor == "self" else "a colleague "
                                action = slot["action"]
                                inflected = (
                                    action
                                    if actor == "self"
                                    else "verifies"
                                    if action == "verify"
                                    else action + "s"
                                )
                                astart = len(text)
                                text += inflected + " "
                                ostart = len(text)
                                text += slot["object"]
                                report["triggers"].append(
                                    {
                                        "action": action,
                                        "action_span": [
                                            astart,
                                            astart + len(inflected),
                                        ],
                                        "object": slot["object"],
                                        "object_span": [ostart, len(text)],
                                        "actor": actor,
                                        "polarity": 1,
                                        "domain": slot["target"],
                                        "relevant": slot["bits"][0] == 1,
                                    }
                                )
                            text += "."
                            report["projection_text"] = text[start:]
                            reports.append(report)
                            if ri == 0:
                                text += " "
                        rows.append(
                            {
                                "material_id": f"f{family:02d}_s{frame}_{actor}_m{mask}",
                                "cube_id": f"f{family:02d}_s{frame}_{actor}",
                                "family": family,
                                "role": role,
                                "frame": frame,
                                "actor": actor,
                                "mask": mask,
                                "kind": "factorial_joint_law",
                                "text": text,
                                "expected_reports": reports,
                                "human_author_id": None,
                            }
                        )
    if len(rows) != spec["new_materials"]:
        raise ValueError("declared factorial material count changed")
    return rows


def prepare(out):
    """Freeze code, split, input addresses and scoring choices before calls."""
    if (out / "04/summary.json").exists():
        raise ValueError("completed packet frozen")
    spec = json.loads(SPEC.read_text())
    source = ROOT / spec["source_packet"]
    source_hash = check_source(source)
    bank = material_bank(read_bank(source / "00/singles.jsonl"), spec)
    paths = [
        Path(__file__),
        SPEC,
        ROOT / "grain_research/construction/math_joint_law.py",
        ROOT / "grain_research/construction/math_state_exposure.py",
        ROOT / "grain_research/construction/math_observation_dag.py",
        ROOT / "scripts/run_grain_math_state_exposure.py",
        ROOT / "scripts/run_grain_math_response.py",
        ROOT / "scripts/run_grain_math_binding.py",
    ]
    signature = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    content = "".join(json.dumps(r) + "\n" for r in bank)
    protocol = out / "00/protocol_before_run.json"
    contract = {
        "spec": spec,
        "signature": signature,
        "source_manifest": source_hash,
        "materials": len(bank),
        "cubes": len(bank) // 8,
        "frozen_before_calls": True,
        "gold_used_for_extraction": False,
        "observer_target_refits": 0,
        "role_units": "disjoint lexical-pair families, not independent human authors",
    }
    if protocol.exists():
        if (
            json.loads(protocol.read_text()) != contract
            or (out / "00/documents.jsonl").read_text() != content
        ):
            raise ValueError("joint-law frozen protocol changed")
    else:
        for p in paths:
            dest = out / "00/code" / p.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
        (out / "00/documents.jsonl").write_text(content)
        write_json(
            out / "00/reader_registry.json",
            json.loads((source / "00/reader_registry.json").read_text()),
        )
        write_json(protocol, contract)
        write_json(out / "00/support_witness.json", J.support_witness())
        write_csv(
            out / "00/family_split.csv",
            [
                {"family": f, "role": next(r["role"] for r in bank if r["family"] == f)}
                for f in range(16)
            ],
        )
    print(f"HM10A frozen {len(bank)} texts; 8/4/4 lexical-pair families", flush=True)
    return source, bank, spec


def indices(out, source, bank, parsed):
    """Address raw documents and extracted views before any target readout."""
    old = {
        r["text"]: i for i, r in enumerate(read_bank(source / "00/material_bank.jsonl"))
    }
    fresh, lookup = [], {}

    def ref(text):
        if text in old:
            return ["source", old[text]]
        if text not in lookup:
            lookup[text] = len(fresh)
            fresh.append({"text": text, "kind": "joint_law_view"})
        return ["new", lookup[text]]

    def decorate(reports):
        return [
            {
                **r,
                "whole_ref": ref(r["projection_text"]),
                "triggers": [
                    {
                        **t,
                        "action_ref": ref("Action: " + t["action"] + "."),
                        "object_ref": ref("Object: " + t["object"] + ".")
                        if t["object"] is not None
                        else None,
                    }
                    for t in r["triggers"]
                ],
            }
            for r in reports
        ]

    indexed = []
    for row, data in zip(bank, parsed, strict=True):
        declared = [
            {
                **r,
                "report_id": f"declared_{ri}",
                "status": "ok",
                "triggers": [
                    {**t, "node_id": f"declared_{ri}_{ti}", "status": "ok"}
                    for ti, t in enumerate(r["triggers"])
                ],
            }
            for ri, r in enumerate(row["expected_reports"])
        ]
        indexed.append(
            {
                "document_ref": ref(row["text"]),
                "observed": decorate(data["reports"]),
                "declared": decorate(declared),
            }
        )
    content = "".join(json.dumps(r) + "\n" for r in fresh)
    path = out / "00/material_bank.jsonl"
    if path.exists() and path.read_text() != content:
        raise ValueError("joint-law encoding bank changed")
    path.write_text(content)
    write_json(out / "01/observation_indices.json", indexed)
    return indexed


def score(out, source, bank, indexed, audit):
    """Retain whole-observer, report-marginal and coexposure quantities separately."""
    rows, conditions = [], []
    for reader in MODELS:
        old = np.load(source / f"01B/embeddings/{reader}.npy").astype(float)
        new = np.load(out / f"01B/embeddings/{reader}.npy").astype(float)
        models = [
            load_model(source / f"02/{reader}_{target}.npz") for target in S.TARGETS
        ]
        h = vectors([r["document_ref"] for r in indexed], old, new)
        whole = np.column_stack(
            [L.affine_score(m["whole_factors"], h)[:, 0] for m in models]
        )
        for i, (material, entry) in enumerate(zip(bank, indexed, strict=True)):
            result = {
                "reader": reader,
                **{
                    k: material[k]
                    for k in (
                        "material_id",
                        "cube_id",
                        "family",
                        "role",
                        "frame",
                        "actor",
                        "mask",
                    )
                },
                "address_complete": audit[i]["complete"],
                "whole_C": float(whole[i, 0]),
                "whole_V": float(whole[i, 1]),
            }
            for arm in ("observed", "declared"):
                reports, capacities = entry[arm], []
                # This gate consults actual parser status only, not declared addresses.
                eligible = bool(reports) and all(
                    r["status"] == "ok" and r["experiencer"] == "self" for r in reports
                )
                for ri, report in enumerate(reports):
                    tt = report["triggers"]
                    if any(t["object_ref"] is None for t in tt):
                        eligible = False
                        continue
                    a = vectors([t["action_ref"] for t in tt], old, new)
                    o = vectors([t["object_ref"] for t in tt], old, new)
                    kk = np.column_stack(
                        [B.score_kernel(m["auto_tensor"], a, o) for m in models]
                    )
                    capacities.append(S.report_capacity(kk, report))
                    for ti, t in enumerate(tt):
                        conditions.append(
                            {
                                "reader": reader,
                                "material_id": material["material_id"],
                                "arm": arm,
                                "report": ri,
                                "trigger": ti,
                                "actor": t["actor"],
                                "C": float(kk[ti, 0]),
                                "V": float(kk[ti, 1]),
                            }
                        )
                result[arm + "_report_mass"] = len(reports) if eligible else None
                for name in ("C", "V"):
                    qi = 0 if name == "C" else 1
                    result[arm + "_" + name] = (
                        float(np.mean([r["coverage"][qi] for r in capacities]))
                        if eligible
                        else None
                    )
                result[arm + "_joint"] = (
                    float(np.mean([r["coexposure"] for r in capacities]))
                    if eligible
                    else None
                )
                result[arm + "_hard_joint"] = (
                    float(np.mean([r["hard_coexposure"] for r in capacities]))
                    if eligible
                    else None
                )
            rows.append(result)
        write_json(
            out / f"02/{reader}_observer_units.json",
            {
                "targets": list(S.TARGETS),
                "whole_unit": "frozen HM03 whole_factors relevance; not self-qualified emotion score",
                "report_unit": "eligible report average; trigger count is not the denominator",
                "source_target_refits": 0,
            },
        )
    write_csv(out / "03/condition_scores.csv", conditions)
    write_csv(out / "03/document_readouts.csv", rows)
    return rows


def analyze(out, source, bank, indexed, rows, spec):
    """Fit source-only joint corrections and reveal all cold-cell outcomes."""
    metrics = [
        "whole_C",
        "whole_V",
        *[
            arm + "_" + q
            for arm in ("declared", "observed")
            for q in ("C", "V", "joint", "hard_joint")
        ],
    ]
    groups = defaultdict(dict)
    for row in rows:
        groups[row["reader"], row["cube_id"]][row["mask"]] = row
    coefficients, energy_rows, tables, vector_rows = [], [], [], []
    for reader in MODELS:
        new = np.load(out / f"01B/embeddings/{reader}.npy").astype(float)
        old = np.load(source / f"01B/embeddings/{reader}.npy").astype(float)
        hx = vectors([r["document_ref"] for r in indexed], old, new)
        lookup = {r["material_id"]: i for i, r in enumerate(bank)}
        vector_mobius = []
        for (rr, cube), cells in groups.items():
            if rr != reader or set(cells) != set(range(8)):
                continue
            template = cells[0]
            metadata = {
                k: template[k]
                for k in ("reader", "cube_id", "family", "role", "frame", "actor")
            }
            hs = np.array([hx[lookup[cells[m]["material_id"]]] for m in range(8)])
            vm = J.mobius(hs)
            vector_mobius.append(vm)
            vector_rows.append(
                {
                    **metadata,
                    **{
                        f"mask_{m}_norm": float(np.linalg.norm(vm[m])) for m in range(8)
                    },
                }
            )
            for metric in metrics:
                f = [cells[m][metric] for m in range(8)]
                if any(value is None for value in f):
                    continue
                f = np.array(f)
                mf = J.mobius(f)
                for mask in range(8):
                    coefficients.append(
                        {
                            **metadata,
                            "metric": metric,
                            "mask": mask,
                            "coefficient": float(mf[mask]),
                        }
                    )
                for reference, p in (
                    ("uniform_product", spec["factor_probabilities"]),
                    ("biased_product_diagnostic", spec["alternate_reference_only"]),
                ):
                    ff = J.product_fourier(f, p)
                    for mask, energy in enumerate(ff["energies"]):
                        energy_rows.append(
                            {
                                **metadata,
                                "metric": metric,
                                "reference": reference,
                                "mask": mask,
                                "energy": float(energy),
                                "variance": ff["variance"],
                                "reconstruction_error": ff["reconstruction_error"],
                                "gram_error": ff["gram_error"],
                                "variance_closure_error": ff["variance_closure_error"],
                            }
                        )
                for incidence in range(2):
                    values = f[incidence * 4 : incidence * 4 + 4]
                    tables.append(
                        {
                            **metadata,
                            "metric": metric,
                            "incidence": incidence,
                            **{f"y{m}": float(values[m]) for m in range(4)},
                            "interaction": float(
                                values[3] - values[1] - values[2] + values[0]
                            ),
                        }
                    )
        np.save(out / f"02/{reader}_vector_mobius.npy", np.array(vector_mobius))
    predictions, fit_rows, summaries = [], [], []
    for reader in MODELS:
        for metric in metrics:
            selected = [
                r for r in tables if r["reader"] == reader and r["metric"] == metric
            ]
            if not any(r["role"] == "source" for r in selected):
                continue
            fitted = J.fit_source_interactions(selected)
            fit_rows.append(
                {
                    "reader": reader,
                    "metric": metric,
                    "key": "global",
                    "mean": fitted["global"],
                    "source_rows": fitted["source_rows"],
                    "source_families": fitted["source_families"],
                }
            )
            for key, values in fitted["conditional"].items():
                fit_rows.append(
                    {"reader": reader, "metric": metric, "key": list(key), **values}
                )
            for r in selected:
                if r["role"] == "source":
                    continue
                key = (r["frame"], r["actor"], r["incidence"])
                shifted = ((r["frame"] + 1) % 4, r["actor"], 1 - r["incidence"])
                corrections = {
                    "additive": 0.0,
                    "source_global": fitted["global"],
                    "source_conditional": fitted["conditional"]
                    .get(key, {})
                    .get("mean"),
                    "source_shifted_key_control": fitted["conditional"]
                    .get(shifted, {})
                    .get("mean"),
                }
                base = r["y1"] + r["y2"] - r["y0"]
                for arm, correction in corrections.items():
                    predictions.append(
                        {
                            **{
                                k: r[k]
                                for k in (
                                    "reader",
                                    "cube_id",
                                    "family",
                                    "role",
                                    "frame",
                                    "actor",
                                    "metric",
                                    "incidence",
                                )
                            },
                            "arm": arm,
                            "additive": base,
                            "source_correction": correction,
                            "actual_joint": r["y3"],
                            "predicted_joint": base + correction
                            if correction is not None
                            else None,
                            "squared_error": (base + correction - r["y3"]) ** 2
                            if correction is not None
                            else None,
                            "additive_squared_error": (base - r["y3"]) ** 2,
                        }
                    )
    summary_keys = sorted(
        {(r["reader"], r["metric"], r["role"], r["arm"]) for r in predictions}
    )
    for key in summary_keys:
        rr = [
            r
            for r in predictions
            if tuple(r[k] for k in ("reader", "metric", "role", "arm")) == key
        ]
        scored = [r for r in rr if r["squared_error"] is not None]
        delta = [r["additive_squared_error"] - r["squared_error"] for r in scored]
        summaries.append(
            {
                **dict(zip(("reader", "metric", "role", "arm"), key, strict=True)),
                "rows": len(rr),
                "scored": len(scored),
                "rmse": float(np.sqrt(np.mean([r["squared_error"] for r in scored])))
                if scored
                else None,
                "additive_rmse": float(
                    np.sqrt(np.mean([r["additive_squared_error"] for r in scored]))
                )
                if scored
                else None,
                "mse_improvement": float(np.mean(delta)) if scored else None,
                "paired_family_ci": J.paired_family_interval(
                    [r["family"] for r in scored],
                    delta,
                    spec["bootstrap"]["seed"],
                    spec["bootstrap"]["draws"],
                )
                if scored
                else None,
            }
        )
    write_csv(out / "03/mobius_coefficients.csv", coefficients)
    write_csv(out / "03/fourier_energies.csv", energy_rows)
    write_csv(out / "03/vector_interactions.csv", vector_rows)
    write_csv(out / "03/four_corner_tables.csv", tables)
    write_csv(out / "02/source_joint_means.csv", fit_rows)
    write_csv(out / "03/cold_joint_predictions.csv", predictions)
    write_csv(out / "03/completion_summary.csv", summaries)
    return summaries, vector_rows, coefficients


def main():
    """Execute one finite new packet with resumable offline observations."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT)
    p.add_argument("--device", default="mps")
    args = p.parse_args()
    source, bank, spec = prepare(args.output)
    parsed = parse(args.output, bank)
    indexed = indices(args.output, source, bank, parsed)
    audit = coverage_audit(bank, indexed)
    write_csv(args.output / "01/coverage_audit.csv", audit)
    for reader in MODELS:
        encode(args.output, reader, args.device)
    rows = score(args.output, source, bank, indexed, audit)
    summaries, vector_rows, coefficients = analyze(
        args.output, source, bank, indexed, rows, spec
    )
    by_cube = defaultdict(dict)
    for r in bank:
        by_cube[r["cube_id"]][r["mask"]] = r
    bags = []
    for cube, cells in by_cube.items():
        for m in range(4):
            a, b = cells[m], cells[m + 4]
            match = Counter(re.findall(r"\w+", a["text"].lower())) == Counter(
                re.findall(r"\w+", b["text"].lower())
            )
            bags.append(
                {
                    "cube_id": cube,
                    "mask": m,
                    "same_word_bag": match,
                    "same_raw_text": a["text"] == b["text"],
                }
            )
    write_csv(args.output / "03/incidence_wordbag_controls.csv", bags)
    summary = {
        "packet": "HM-10A",
        "materials": len(bank),
        "cubes": len(by_cube),
        "source_validation_application_families": [8, 4, 4],
        "complete_documents": sum(r["complete"] for r in audit),
        "failures": [r for r in audit if not r["complete"]],
        "same_word_incidence_pairs": sum(r["same_word_bag"] for r in bags),
        "incidence_pairs": len(bags),
        "encoder_inputs_each_reader": len(
            read_bank(args.output / "00/material_bank.jsonl")
        ),
        "completion_summary": summaries,
        "vector_interaction_summary": [
            {
                "reader": reader,
                "cubes": sum(r["reader"] == reader for r in vector_rows),
                "joint_vector_norm_mean": float(
                    np.mean(
                        [r["mask_3_norm"] for r in vector_rows if r["reader"] == reader]
                    )
                ),
                "incidence_joint_vector_norm_mean": float(
                    np.mean(
                        [r["mask_7_norm"] for r in vector_rows if r["reader"] == reader]
                    )
                ),
            }
            for reader in MODELS
        ],
        "mobius_rows": len(coefficients),
        "source_observer_refits": 0,
        "source_joint_mean_fits_only": True,
        "individual_response_parameters_fitted": False,
        "scope": "actual frozen observer contrasts and lexical-family masked-cell completion, not psychological author response/intensity",
    }
    check_source(source)
    write_json(args.output / "04/summary.json", summary)
    write_json(
        args.output / "04/artifact_hashes.json",
        {
            str(f.relative_to(args.output)): sha(f)
            for f in sorted(args.output.rglob("*"))
            if f.is_file() and f.name != "artifact_hashes.json"
        },
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in {"failures", "completion_summary"}
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
