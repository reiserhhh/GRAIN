#!/usr/bin/env python3
"""Finite conditional target/reference construction on SAME cached author vectors."""

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
from grain_research.construction import math_context_relation as C
from grain_research.construction import math_profile_reader as M
from grain_research.construction import math_vector_lift as V
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_profile_reader_hm182.json"
DEFAULT = ROOT / "results/grain_math_response_v1/conditional_profile_reader_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_PROFILE_READER_PLAN_HM182.md",
    "grain_research/construction/math_profile_reader.py",
    "scripts/run_grain_profile_reader_hm182.py",
    "scripts/score_grain_profile_reader_cold.py",
    "tests/test_grain_profile_reader.py",
)


def cosine(x, y):
    """Keep undefined zero directions missing,without sign alignment."""
    a, b = np.linalg.norm(x), np.linalg.norm(y)
    return float(np.dot(x, y) / (a * b)) if a > 0 and b > 0 else None


def prepare(out):
    """Seal source/data/finite protocol before reading outcome comparisons."""
    spec = R.read(ROOT / CONFIG)
    par = ROOT / spec["parent"]
    native = ROOT / spec["native_parent"]
    prior = R.check(
        par, ("compiled_seal.json", "scored_seal.json", "verification_seal.json")
    )
    R.check(native, ("scored_seal.json", "analysis_seal.json"))
    files = [
        str((p / name).relative_to(ROOT))
        for p, names in (
            (
                par,
                (
                    "manifest.json",
                    "prepared_seal.json",
                    "compiled_seal.json",
                    "scored_seal.json",
                    "verification_seal.json",
                    "observations.npz",
                    "cold_input.jsonl",
                ),
            ),
            (
                native,
                (
                    "manifest.json",
                    "prepared_seal.json",
                    "scored_seal.json",
                    "analysis_seal.json",
                    "mechanism_accounts.npz",
                    "cold_input.jsonl",
                ),
            ),
        )
        for name in names
    ]
    z = R.prepared(par)
    if (
        z["source_coordinates"].shape != (386, 2, 12, 15)
        or z["targets"].shape != (386, 5)
        or set(z["source"]) & set(z["panel"])
    ):
        raise ValueError("Changed source/panel identity or feature family")
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": spec,
            "code": prior["code"] | {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {p: R.digest(ROOT / p) for p in files},
            "original_scope_complete": False,
            "historical_cohort_not_lockbox": True,
        },
    )
    R.archive(out / "observations.npz", **z)
    with (out / "cold_input.jsonl").open("xb") as sink:
        sink.write((par / "cold_input.jsonl").read_bytes())
    R.write(
        out / "quality.json",
        {
            "source_authors": 386,
            "panel_authors": 24,
            "inherited_binary_labels_not_recut": True,
            "source_raw_application_labels_new_neural_reads": 0,
            "all_coordinates": 15,
            "feature_terms": 4096,
            "triples_per_author_view": 64,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "cold_input.jsonl", "quality.json"),
    )
    print(json.dumps(R.read(out / "quality.json")))


def compile_reader(out):
    """Make all conditional/marginal fields without application outcomes or fits."""
    R.check(out)
    z = R.prepared(out)
    began = time.perf_counter()
    x = np.array(
        [
            [
                V.subset_feature(author, np.arange(12))[0]
                for author in z["source_coordinates"][:, v]
            ]
            for v in range(2)
        ]
    )
    made = [M.construct(x[v], z["targets"]) for v in range(2)]
    a = {k: np.stack([r[k] for r in made]) for k in made[0]}
    R.archive(out / "source_fields.npz", features=x, **a)
    b = {
        "version": M.VERSION,
        "code_hash": M.code_hash(),
        "targets": list(M.TARGETS),
        "components": list(M.COMPONENTS),
        "fields": a["fields"].tolist(),
        "edges": a["edges"].tolist(),
        "omega": a["omega"][0].tolist(),
        "interpretation": "source_descriptive_profile_contrasts_not_probability_or_causal_trait",
    }
    M.validate_bundle(b)
    R.write(out / "reader_bundle.json", b)
    rows = []
    balance_error = 0.0
    for v, q in np.ndindex(2, 5):
        fields = a["fields"][v, q]
        n = a["counts"][v, q]
        w = a["weights"][v, q]
        key = M.profile_codes(z["targets"], q)
        ess = []
        for g in (0, 1):
            mask = z["targets"][:, q] == g
            ww = np.abs(w[mask])
            balance_error = max(
                balance_error,
                float(
                    np.max(
                        np.abs(
                            np.bincount(key[mask], weights=ww, minlength=16)
                            - a["omega"][v, q]
                        )
                    )
                ),
            )
            ess.append(float(ww.sum() ** 2 / (ww @ ww)))
        cross = 2 * sum(
            float(fields[j] @ fields[k])
            for j, k in itertools.combinations(range(1, 4), 2)
        )
        rows.append(
            {
                "source_view": v,
                "target": M.TARGETS[q],
                "marginal_matched_cosine": cosine(fields[0], fields[1]),
                "marginal_norm": float(np.linalg.norm(fields[0])),
                "matched_norm": float(np.linalg.norm(fields[1])),
                "reference_shift_norm": float(np.linalg.norm(fields[2])),
                "composition_norm": float(np.linalg.norm(fields[3])),
                "signed_three_term_cross": cross,
                "low_ESS": ess[0],
                "high_ESS": ess[1],
                "min_group_count": int(n.min()),
            }
        )
    R.table(out / "field_diagnostics.csv", rows)
    erows = []
    for v, q, bindex in np.ndindex(2, 5, 16):
        erows.append(
            {
                "source_view": v,
                "target": M.TARGETS[q],
                "other_profile": bindex,
                "count0": int(a["counts"][v, q, 0, bindex]),
                "count1": int(a["counts"][v, q, 1, bindex]),
                "omega": float(a["omega"][v, q, bindex]),
                "edge_matched_cosine": cosine(
                    a["edges"][v, q, bindex], a["fields"][v, q, 1]
                ),
                "cross_view_edge_cosine": cosine(
                    a["edges"][0, q, bindex], a["edges"][1, q, bindex]
                ),
            }
        )
    R.table(out / "all_edge_diagnostics.csv", erows)
    gram = []
    for q, r, j, k in np.ndindex(5, 5, 4, 4):
        gram.append(
            {
                "target_view0": M.TARGETS[q],
                "target_view1": M.TARGETS[r],
                "component_view0": M.COMPONENTS[j],
                "component_view1": M.COMPONENTS[k],
                "cosine": cosine(a["fields"][0, q, j], a["fields"][1, r, k]),
            }
        )
    R.table(out / "all_cross_view_fields.csv", gram)
    error = float(np.max(np.abs(a["fields"][:, :, 0] - a["fields"][:, :, 1:].sum(2))))
    if (
        max(error, balance_error)
        > R.read(out / "manifest.json")["config"]["comparison_tolerance"]
    ):
        raise ValueError("Fixed reference/three-term contract failed")
    receipt = {
        "seconds": time.perf_counter() - began,
        "source_authors": 386,
        "conditional_edges": 160,
        "marginal_and_component_fields": 40,
        "profile_balance_error": balance_error,
        "decomposition_error": error,
        "all_source_weights_nonzero": bool(np.all(a["weights"] != 0)),
        "no_fit_accuracy_or_new_neural": True,
        "original_scope_complete": False,
    }
    R.write(out / "construction_receipt.json", receipt)
    R.seal(
        out,
        "compiled_seal.json",
        (
            "source_fields.npz",
            "reader_bundle.json",
            "field_diagnostics.csv",
            "all_edge_diagnostics.csv",
            "all_cross_view_fields.csv",
            "construction_receipt.json",
        ),
    )
    print(json.dumps(receipt))


def score(out):
    """Cold score SAME historical evidence before native contact evaluation."""
    R.check(out, ("compiled_seal.json",))
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_profile_reader_cold.py"),
            "--bundle",
            str(out / "reader_bundle.json"),
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
        raise ValueError("Cold scope changed")
    summary = R.read(out / "construction_receipt.json") | {
        "cold_records": 240,
        "component_values": 9600,
        "edge_values": 38400,
        "missing_triple_inputs": sum(not r["triples"] for r in records),
    }
    R.write(out / "cold_summary.json", summary)
    R.seal(out, "scored_seal.json", ("cold_output.jsonl", "cold_summary.json"))
    print(json.dumps(summary))


def analyze(out):
    """Keep all prescribed native contacts,without picking the strongest field."""
    spec = R.check(out, ("compiled_seal.json", "scored_seal.json"))["config"]
    native = ROOT / spec["native_parent"]
    R.check(native, ("scored_seal.json", "analysis_seal.json"))
    legacy = {
        r["author_key"]: r
        for r in map(json.loads, (native / "cold_input.jsonl").read_text().splitlines())
    }
    count = 0
    for row in map(json.loads, (out / "cold_input.jsonl").read_text().splitlines()):
        prior = legacy["1/" + row["frame_key"]]
        if [(e["event_id"], e["slot"]) for e in prior["events"]] != [
            (e["event_id"], e["slot"]) for e in row["events"]
        ]:
            raise ValueError("Native contact does not share exact evidence")
        count += 1
    with np.load(native / "mechanism_accounts.npz", allow_pickle=False) as f:
        endpoints = np.concatenate(
            (f["native_ordinal_actions"][..., None], f["native_log_odds_actions"]),
            axis=2,
        )
    readings = np.zeros((24, 2, 5, 2, 5, 4))
    for row in map(json.loads, (out / "cold_output.jsonl").read_text().splitlines()):
        readings[tuple(map(int, row["frame_key"].split("/")))] = row["retained_fields"]
    actions = (
        readings[:, :, 2] - readings[:, :, 1] - readings[:, :, 3] + readings[:, :, 4]
    )
    rows = []
    for sv, q, component, endpoint in np.ndindex(2, 5, 4, 10):
        for group in ("author_mean", "view0", "view1"):
            x, y = actions[:, :, sv, q, component], endpoints[:, :, endpoint]
            if group == "author_mean":
                x, y = x.mean(1), y.mean(1)
            else:
                v = int(group[-1])
                x, y = x[:, v], y[:, v]
            rows.append(
                {
                    "source_view": sv,
                    "source_target": M.TARGETS[q],
                    "component": M.COMPONENTS[component],
                    "native_query": "N",
                    "endpoint": "ordinal" if endpoint == 0 else f"digit{endpoint}_vs0",
                    "group": group,
                    **C.signed_agreement(x, y),
                }
            )
    error = float(
        np.max(np.abs(actions[:, :, :, :, 0] - actions[:, :, :, :, 1:].sum(4)))
    )
    if error > spec["comparison_tolerance"]:
        raise ValueError("Cold action decomposition failed")
    R.table(out / "all_native_contacts.csv", rows)
    R.archive(
        out / "native_contacts.npz",
        readings=readings,
        actions=actions,
        native_endpoints=endpoints,
    )
    summary = R.read(out / "cold_summary.json") | {
        "same_native_frames": count,
        "all_native_links": len(rows),
        "action_decomposition_error": error,
        "N_ordinal_author_mean": [
            r
            for r in rows
            if r["source_target"] == "N"
            and r["endpoint"] == "ordinal"
            and r["group"] == "author_mean"
        ],
        "no_accuracy_causal_or_independent_axis_promotion": True,
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analysis_seal.json",
        ("all_native_contacts.csv", "native_contacts.npz", "summary.json"),
    )
    print(json.dumps(summary))


def audit(out):
    """Alternate literal tensor/author weighting/cold replay;not a blind audit."""
    spec = R.check(
        out, ("compiled_seal.json", "scored_seal.json", "analysis_seal.json")
    )["config"]
    z = R.prepared(out)
    with np.load(out / "source_fields.npz", allow_pickle=False) as f:
        a = {k: f[k] for k in f.files}
    direct = []
    triples = 0
    for v in range(2):
        features = []
        for author in z["source_coordinates"][:, v]:
            phi = np.zeros(4096)
            for i, j, k in itertools.product(range(4), range(4, 8), range(8, 12)):
                phi += (
                    np.kron(
                        np.kron(np.r_[1, author[i]], np.r_[1, author[j]]),
                        np.r_[1, author[k]],
                    )
                    / 64
                )
                triples += 1
            features.append(phi)
        direct.append(features)
    direct = np.asarray(direct)
    feature_error = float(np.max(np.abs(direct - a["features"])))
    weight_error = 0.0
    for q in range(5):
        b = M.profile_codes(z["targets"], q)
        n = a["counts"][0, q]
        e = n[1] / n.sum(0)
        raw = np.where(z["targets"][:, q] == 1, 1 - e[b], -e[b])
        w = raw / raw[raw > 0].sum()
        for v in range(2):
            weight_error = max(
                weight_error,
                float(np.max(np.abs(w @ direct[v] - a["fields"][v, q, 1]))),
            )
    bundle = R.read(out / "reader_bundle.json")
    fw = np.asarray(bundle["fields"])
    ew = np.asarray(bundle["edges"])
    cold = {
        r["frame_key"]: r
        for r in map(json.loads, (out / "cold_output.jsonl").read_text().splitlines())
    }
    cold_error = 0.0
    cold_triples = 0
    for r in map(json.loads, (out / "cold_input.jsonl").read_text().splitlines()):
        roles = [[e for e in r["events"] if e["slot"] // 4 == k] for k in range(3)]
        total = np.zeros(4096)
        for i, j, k in itertools.product(*roles):
            total += (
                np.kron(
                    np.kron(np.r_[1, i["coordinates"]], np.r_[1, j["coordinates"]]),
                    np.r_[1, k["coordinates"]],
                )
                / 64
            )
            cold_triples += 1
        actual = cold[r["frame_key"]]
        cold_error = max(
            cold_error,
            float(
                np.max(
                    np.abs(
                        np.einsum("vqck,k->vqc", fw, total) - actual["retained_fields"]
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        np.einsum("vqbk,k->vqb", ew, total) - actual["retained_edges"]
                    )
                )
            ),
        )
        if not actual["triples"] and (
            actual["conditional_fields"] is not None
            or actual["conditional_edges"] is not None
        ):
            raise ValueError("Missing opportunity relabeled a trait")
    result = {
        "source_triples": triples,
        "cold_triples": cold_triples,
        "source_feature_error": feature_error,
        "opposite_group_weight_error": weight_error,
        "cold_literal_error": cold_error,
        "same_agent_not_blind": True,
        "original_scope_complete": False,
    }
    if max(feature_error, weight_error, cold_error) > spec["comparison_tolerance"]:
        raise ValueError("Alternate finite audit failed")
    R.write(out / "verification.json", result)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(result))


def main():
    """Execute one named immutable phase;never restart a failed scientific directory."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("prepare", "compile", "score", "analyze", "audit"))
    p.add_argument("--out", type=Path, default=DEFAULT)
    args = p.parse_args()
    {
        "prepare": prepare,
        "compile": compile_reader,
        "score": score,
        "analyze": analyze,
        "audit": audit,
    }[args.phase](args.out)


if __name__ == "__main__":
    main()
