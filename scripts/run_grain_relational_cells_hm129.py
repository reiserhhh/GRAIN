#!/usr/bin/env python3
"""Fixed neutral-source relational transfer with a sealed numeric cold scorer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_relational_cells as M

MODEL = (
    ROOT
    / "results/grain_vector_mechanisms/local_observers/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)
READER = ROOT / "results/grain_math_response_v1/expression_response_r1/00/reader.json"
DEFAULT = ROOT / "results/grain_math_response_v1/relational_cells_r2"
SYSTEM = "Read the following records."
FORWARD_CAP, SECONDS_CAP = 84, 240
CODE = (
    ROOT / "grain_research/construction/math_relational_cells.py",
    Path(__file__).resolve(),
    ROOT / "scripts/score_grain_relational_cells_cold.py",
    ROOT / "grain_research/vector_mechanisms/trace.py",
    ROOT / "docs/GRAIN_RELATIONAL_CELL_TRANSFER_PLAN_HM129.md",
    ROOT / "tests/test_grain_relational_cells.py",
)


def digest(path):
    """Hash a frozen file without loading scientific content into fit inputs."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    """Load an explicit JSON stage artifact."""
    return json.loads(Path(path).read_text())


def write(path, value):
    """Atomically checkpoint runtime metadata, refusing nonfinite JSON."""
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def save(path, **arrays):
    """Write each numeric stage once, retaining earlier checkpoints."""
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def check_seal(output, name):
    """Verify every sealed stage dependency before downstream consumption."""
    seals = read(output / name)
    for path, expected in seals.items():
        if digest(path) != expected:
            raise RuntimeError(f"Frozen dependency changed: {path}")
    return len(seals)


def prepare(output):
    """Separate source labels, cold bodies and evaluation truth before collection."""
    from transformers import AutoTokenizer

    output.mkdir(parents=True, exist_ok=False)
    (output / "units").mkdir()
    rows, truth = M.material_bank()
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    for row in rows:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": row["body"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(
            prompt, add_special_tokens=False, return_offsets_mapping=True
        )
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        ids = ids["input_ids"] if isinstance(ids, Mapping) else ids
        if (
            ids != encoded["input_ids"]
            or len(ids) + 16 > 512
            or prompt.count(row["body"]) != 1
        ):
            raise ValueError("Exact bounded chat-template addressing required")
        origin = prompt.index(row["body"])
        positions = []
        for start, end in M.record_intervals(row["body"]):
            eligible = [
                i
                for i, (a, b) in enumerate(encoded["offset_mapping"])
                if b > a >= origin + start and b <= origin + end
            ]
            if not eligible:
                raise ValueError("Record endpoint has no complete token")
            positions.append(eligible[-1])
        row["ids"], row["positions"] = ids, positions
        row["endpoint_tokens"] = [
            tokenizer.convert_ids_to_tokens(ids[p]) for p in positions
        ]
    write(output / "source_inputs.json", [r for r in rows if r["role"] == "source"])
    write(output / "cold_inputs.json", [r for r in rows if r["role"] != "source"])
    write(output / "source_labels.json", [t for t in truth if t["id"] < 24])
    write(output / "evaluation_truth.json", truth)
    reader = read(READER)
    save(
        output / "target_covectors.npz",
        directions=np.asarray(reader["directions"], float),
    )
    write(
        output / "design.json",
        {
            "source_worlds": 24,
            "cold_fact_worlds": 12,
            "cold_behaviour_worlds": 48,
            "total_records": 504,
            "system": SYSTEM,
            "targets": reader["targets"],
            "model": str(MODEL),
            "dtype": "float32",
            "device": "mps",
            "attention": "eager",
            "structural_layer": 15,
            "value_layer": 17,
            "pls_components": 2,
            "maximum_forwards": FORWARD_CAP,
            "maximum_model_seconds": SECONDS_CAP,
            "no_weight_updates": True,
            "no_native_answer_or_query": True,
            "explicit_roster_record_interface": True,
            "human_validity_claim": False,
            "model_weights_locally_pinned_directory_not_rehashed": True,
        },
    )
    files = (
        list(CODE)
        + list(output.glob("*.json"))
        + [output / "target_covectors.npz", READER]
    )
    files += [p for p in MODEL.iterdir() if p.suffix == ".json"]
    write(output / "input_seal.json", {str(p): digest(p) for p in files})
    write(
        output / "accounting.json",
        {"phase": "PREPARED", "calls": 0, "model_seconds": 0.0, "completed_ids": []},
    )


def collect(output, stage):
    """Capture bounded pre-answer endpoints once; never read evaluation truth."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    check_seal(output, "input_seal.json")
    if stage == "cold":
        check_seal(output, "fit_seal.json")
    records = read(output / f"{stage}_inputs.json")
    account = read(output / "accounting.json")
    if any(r["id"] in account["completed_ids"] for r in records):
        raise RuntimeError("No implicit replay/resume of collected observations")
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=512
    )
    config = observer.model.config
    if (config.hidden_size, config.num_attention_heads, config.num_key_value_heads) != (
        1536,
        12,
        2,
    ):
        raise ValueError("Frozen reader/model geometry mismatch")
    observer.sites = {
        "structural": observer.model.model.layers[15],
        "value": observer.model.model.layers[17].self_attn.v_proj,
    }
    for row in records:
        if account["calls"] >= FORWARD_CAP or account["model_seconds"] >= SECONDS_CAP:
            account["phase"] = "STOP_BUDGET"
            write(output / "accounting.json", account)
            raise RuntimeError("Original model budget exhausted; no automatic retry")
        torch.mps.synchronize()
        start = time.perf_counter()
        account["calls"] += 1
        account["inflight_id"] = row["id"]
        write(output / "accounting.json", account)
        try:
            captured, _unused_logits = observer.capture(
                row["ids"], positions=row["positions"]
            )
            torch.mps.synchronize()
        finally:
            account["model_seconds"] += time.perf_counter() - start
            write(output / "accounting.json", account)
        save(
            output / "units" / f"{row['id']:03d}.npz",
            states=captured["structural"],
            values=captured["value"].reshape(6, 2, 128),
        )
        account["completed_ids"].append(row["id"])
        account.pop("inflight_id")
        account["phase"] = f"COLLECTING_{stage.upper()}"
        write(output / "accounting.json", account)
        print(
            json.dumps(
                {
                    "id": row["id"],
                    "calls": account["calls"],
                    "seconds": account["model_seconds"],
                }
            ),
            flush=True,
        )
    account["phase"] = f"{stage.upper()}_COLLECTED"
    write(output / "accounting.json", account)


def numeric_inputs(output, records):
    """Stack numeric-only co-addressed states, excluding labels and bodies."""
    states, values = [], []
    for row in records:
        with np.load(
            output / "units" / f"{row['id']:03d}.npz", allow_pickle=False
        ) as unit:
            states.append(unit["states"])
            values.append(unit["values"])
    return {
        "ids": np.array([r["id"] for r in records]),
        "states": np.stack(states),
        "values": np.stack(values),
    }


def fit(output):
    """Fit source-only structural programs before cold vector collection."""
    check_seal(output, "input_seal.json")
    records, labels = (
        read(output / "source_inputs.json"),
        read(output / "source_labels.json"),
    )
    if read(output / "accounting.json")["completed_ids"] != list(range(24)):
        raise RuntimeError("Exactly source observations required at program freeze")
    data = numeric_inputs(output, records)
    y = np.array([r["roles"] for r in labels]).reshape(-1, 2)
    x = data["states"].reshape(-1, 1536)
    real = M.fit_program(x, y)
    shuffled = np.random.default_rng(12901).permutation(len(y))
    permuted = M.fit_program(x, y[shuffled])
    arrays = {
        f"{arm}_{k}": v
        for arm, p in (("real", real), ("permuted", permuted))
        for k, v in p.items()
    }
    with np.load(output / "target_covectors.npz", allow_pickle=False) as cov:
        arrays["directions"] = cov["directions"]
    arrays["position"] = y.reshape(24, 6, 2).mean(0)
    save(output / "program.npz", **arrays)
    save(output / "source_numeric.npz", **data)
    write(
        output / "fit_metadata.json",
        {
            "source_world_ids": list(range(24)),
            "source_records": 144,
            "components": 2,
            "permutation_seed": 12901,
            "target_label_fit": False,
            "real_iterations": real["iterations"].tolist(),
            "permuted_iterations": permuted["iterations"].tolist(),
            "cold_collection_started": False,
            "unit_hashes": {
                str(p): digest(p) for p in sorted((output / "units").glob("*.npz"))
            },
        },
    )
    files = [
        output / "program.npz",
        output / "fit_metadata.json",
        output / "source_numeric.npz",
    ]
    write(output / "fit_seal.json", {str(p): digest(p) for p in files})


def cold_score(output):
    """Use the same isolated numeric CLI for source replay and cold application."""
    check_seal(output, "fit_seal.json")
    save(
        output / "cold_numeric.npz",
        **numeric_inputs(output, read(output / "cold_inputs.json")),
    )
    for stage in ("source", "cold"):
        command = [
            sys.executable,
            str(ROOT / "scripts/score_grain_relational_cells_cold.py"),
            str(output / "program.npz"),
            str(output / f"{stage}_numeric.npz"),
            str(output / f"{stage}_scored.npz"),
        ]
        subprocess.run(command, check=True, cwd=ROOT)
    files = [
        output / f"{s}_{k}.npz"
        for s in ("source", "cold")
        for k in ("numeric", "scored")
    ]
    write(output / "score_seal.json", {str(p): digest(p) for p in files})


def analyze(output):
    """Evaluate only after cold routing, reporting failures and missing cells."""
    check_seal(output, "score_seal.json")
    truths = {t["id"]: t for t in read(output / "evaluation_truth.json")}
    records = {
        t["id"]: t
        for t in read(output / "source_inputs.json") + read(output / "cold_inputs.json")
    }
    metrics, predictions, routed = [], [], {}
    for stage in ("source", "cold"):
        with np.load(output / f"{stage}_scored.npz", allow_pickle=False) as scored:
            for index, world_id in enumerate(scored["ids"]):
                wid = int(world_id)
                gold_roles = np.array(truths[wid]["roles"])
                readings = scored["readings"][index]
                gold_means, _ = M.cell_ledger(gold_roles, readings)
                for arm in ("real", "permuted", "position"):
                    roles = scored[f"{arm}_roles"][index]
                    means, counts = (
                        scored[f"{arm}_means"][index],
                        scored[f"{arm}_counts"][index],
                    )
                    finite = np.isfinite(means)
                    squared = (means[finite] - gold_means[finite]) ** 2
                    metrics.append(
                        {
                            "id": wid,
                            "domain": records[wid]["role"],
                            "arm": arm,
                            "records": 6,
                            "entity_correct": int(
                                (roles[:, 0] == gold_roles[:, 0]).sum()
                            ),
                            "relation_correct": int(
                                (roles[:, 1] == gold_roles[:, 1]).sum()
                            ),
                            "joint_correct": int(
                                np.all(roles == gold_roles, axis=1).sum()
                            ),
                            "covered_cells": int((counts > 0).sum()),
                            "squared_error_sum": float(squared.sum()),
                            "observed_scalar_count": int(finite.sum()),
                        }
                    )
                    routed[wid, arm] = means
                    for record in range(6):
                        predictions.append(
                            {
                                "id": wid,
                                "domain": records[wid]["role"],
                                "arm": arm,
                                "record": record,
                                "true_entity": int(gold_roles[record, 0]),
                                "true_relation": int(gold_roles[record, 1]),
                                "pred_entity": int(roles[record, 0]),
                                "pred_relation": int(roles[record, 1]),
                                "entity_coordinate": float(
                                    scored[f"{arm}_coordinates"][index, record, 0]
                                ),
                                "relation_coordinate": float(
                                    scored[f"{arm}_coordinates"][index, record, 1]
                                ),
                            }
                        )
                routed[wid, "gold"] = gold_means
    for filename, rows in (
        ("metrics_by_world.csv", metrics),
        ("record_routing.csv", predictions),
    ):
        with (output / filename).open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "status": "FINITE_STRUCTURAL_TRANSFER_NOT_HUMAN_VALIDITY",
        "design": read(output / "design.json"),
        "accounting": read(output / "accounting.json"),
        "domains": {},
        "behaviour_by_person": {},
        "behaviour_by_design": {},
        "matched_assignments": [],
        "full_scope_complete": False,
    }
    for domain in ("source", "cold_fact", "cold_behaviour"):
        summary["domains"][domain] = {}
        for arm in ("real", "permuted", "position"):
            chosen = [r for r in metrics if r["domain"] == domain and r["arm"] == arm]
            total = {
                k: sum(r[k] for r in chosen)
                for k in (
                    "records",
                    "entity_correct",
                    "relation_correct",
                    "joint_correct",
                    "covered_cells",
                    "squared_error_sum",
                    "observed_scalar_count",
                )
            }
            total["cell_rmse_on_observed_only"] = float(
                np.sqrt(total["squared_error_sum"] / total["observed_scalar_count"])
            )
            summary["domains"][domain][arm] = total
    behavioural = [p for p in predictions if p["domain"] == "cold_behaviour"]
    for person in range(3):
        chosen = [
            p for p in behavioural if p["arm"] == "real" and p["true_entity"] == person
        ]
        summary["behaviour_by_person"][str(person)] = {
            "n": len(chosen),
            "correct": sum(p["true_entity"] == p["pred_entity"] for p in chosen),
        }
    for key in ("family", "wording", "header_reverse", "assignment"):
        summary["behaviour_by_design"][key] = {}
        for value in sorted({truths[p["id"]][key] for p in behavioural}):
            summary["behaviour_by_design"][key][str(value)] = {}
            for arm in ("real", "permuted", "position"):
                chosen = [
                    p
                    for p in behavioural
                    if p["arm"] == arm and truths[p["id"]][key] == value
                ]
                summary["behaviour_by_design"][key][str(value)][arm] = {
                    "n": len(chosen),
                    "joint_correct": sum(
                        p["true_entity"] == p["pred_entity"]
                        and p["true_relation"] == p["pred_relation"]
                        for p in chosen
                    ),
                }
    pair_index = {
        (t["family"], t["wording"], t["header_reverse"], t["assignment"]): wid
        for wid, t in truths.items()
        if records[wid]["role"] == "cold_behaviour"
    }
    for (family, wording, header, assignment), a in pair_index.items():
        if assignment != 0:
            continue
        b = pair_index[family, wording, header, 1]
        gold = (routed[b, "gold"][0] - routed[b, "gold"][1]) - (
            routed[a, "gold"][0] - routed[a, "gold"][1]
        )
        row = {
            "ids": [a, b],
            "family": family,
            "wording": wording,
            "header_reverse": header,
            "gold_contrast_rms": float(np.sqrt(np.mean(gold**2))),
        }
        for arm in ("real", "permuted", "position"):
            value = (routed[b, arm][0] - routed[b, arm][1]) - (
                routed[a, arm][0] - routed[a, arm][1]
            )
            finite = np.isfinite(value)
            row[arm] = {
                "observed_scalars": int(finite.sum()),
                "rmse": float(np.sqrt(np.mean((value[finite] - gold[finite]) ** 2)))
                if finite.any()
                else None,
            }
        summary["matched_assignments"].append(row)
    exceeds_controls = all(
        summary["domains"][domain]["real"]["joint_correct"]
        > summary["domains"][domain][arm]["joint_correct"]
        for domain in ("cold_fact", "cold_behaviour")
        for arm in ("permuted", "position")
    )
    all_people = all(
        v["correct"] > v["n"] / 3 for v in summary["behaviour_by_person"].values()
    )
    summary["limited_modular_gate"] = (
        "PASS" if exceeds_controls and all_people else "FAIL"
    )
    summary["gate_components"] = {
        "both_domains_exceed_both_controls": exceeds_controls,
        "each_behaviour_person_above_nominal_one_third": all_people,
        "nominal_threshold_not_statistical_significance": True,
    }
    write(output / "summary.json", summary)
    files = list((output / "units").glob("*.npz")) + [
        output / "summary.json",
        output / "metrics_by_world.csv",
        output / "record_routing.csv",
    ]
    write(output / "result_seal.json", {str(p): digest(p) for p in files})
    print(
        json.dumps(
            {"gate": summary["limited_modular_gate"], "domains": summary["domains"]},
            indent=2,
        ),
        flush=True,
    )


def verify(output):
    """Recompute routing/readouts/cell means with independent numeric loops."""
    hashes = sum(
        check_seal(output, n)
        for n in (
            "input_seal.json",
            "fit_seal.json",
            "score_seal.json",
            "result_seal.json",
        )
    )
    max_readout, max_coordinate, max_cell = 0.0, 0.0, 0.0
    records = 0
    with np.load(output / "program.npz", allow_pickle=False) as program:
        for stage in ("source", "cold"):
            with (
                np.load(output / f"{stage}_numeric.npz", allow_pickle=False) as data,
                np.load(output / f"{stage}_scored.npz", allow_pickle=False) as scored,
            ):
                for index, states in enumerate(data["states"]):
                    values = data["values"][index]
                    ref = np.zeros((6, 6))
                    for e in range(6):
                        for q in range(6):
                            for h in range(12):
                                ref[e, q] += (
                                    values[e, h // 6].astype(float)
                                    @ program["directions"][q, h]
                                )
                    max_readout = max(
                        max_readout,
                        float(np.max(np.abs(ref - scored["readings"][index]))),
                    )
                    for arm in ("real", "permuted", "position"):
                        if arm == "position":
                            coordinates = program["position"]
                        else:
                            c = (
                                program[f"{arm}_coefficients"]
                                / program[f"{arm}_scale"][:, None]
                            )
                            bias = (
                                program[f"{arm}_intercept"]
                                - program[f"{arm}_center"] @ c
                            )
                            coordinates = states.astype(float) @ c + bias
                        max_coordinate = max(
                            max_coordinate,
                            float(
                                np.max(
                                    np.abs(
                                        coordinates
                                        - scored[f"{arm}_coordinates"][index]
                                    )
                                )
                            ),
                        )
                        roles = np.minimum(
                            np.maximum(np.floor(coordinates + 0.5), 0), [2, 1]
                        ).astype(int)
                        if not np.array_equal(roles, scored[f"{arm}_roles"][index]):
                            raise RuntimeError("Independent routing differs")
                        for person in range(3):
                            for condition in range(2):
                                indices = [
                                    e
                                    for e, role in enumerate(roles)
                                    if tuple(role) == (person, condition)
                                ]
                                if (
                                    len(indices)
                                    != scored[f"{arm}_counts"][index, person, condition]
                                ):
                                    raise RuntimeError("Independent cell count differs")
                                mean = scored[f"{arm}_means"][index, person, condition]
                                if not indices:
                                    if not np.isnan(mean).all():
                                        raise RuntimeError("Empty cell silently scored")
                                else:
                                    expected = sum(ref[e] for e in indices) / len(
                                        indices
                                    )
                                    max_cell = max(
                                        max_cell, float(np.max(np.abs(expected - mean)))
                                    )
                    records += 6
    if max(max_readout, max_coordinate, max_cell) > 1e-9:
        raise RuntimeError("Independent arithmetic tolerance failed")
    write(
        output / "verification.json",
        {
            "status": "PASS_NUMERIC_NOT_SCIENCE",
            "frozen_hash_entries": hashes,
            "records": records,
            "readout_max_error": max_readout,
            "coordinate_max_error": max_coordinate,
            "cell_max_error": max_cell,
            "independent_agent_or_causal_test": False,
        },
    )


def main():
    """Run named stages or one finite source-first experiment without retries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "source",
            "fit",
            "cold",
            "score",
            "analyze",
            "verify",
            "run",
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    output = args.output.resolve()
    start = datetime.now(timezone.utc).isoformat()
    try:
        stages = (
            ("prepare", "source", "fit", "cold", "score", "analyze", "verify")
            if args.stage == "run"
            else (args.stage,)
        )
        for stage in stages:
            print(f"STAGE {stage} {datetime.now(timezone.utc).isoformat()}", flush=True)
            if stage in ("source", "cold"):
                collect(output, stage)
            else:
                {
                    "prepare": prepare,
                    "fit": fit,
                    "score": cold_score,
                    "analyze": analyze,
                    "verify": verify,
                }[stage](output)
        print("FINITE_PACKET_DONE_NOT_GLOBAL", flush=True)
    except Exception as error:
        if output.exists():
            write(
                output / "error.json",
                {
                    "started_utc": start,
                    "stage_request": args.stage,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "automatic_retry": False,
                },
            )
        raise


if __name__ == "__main__":
    main()
