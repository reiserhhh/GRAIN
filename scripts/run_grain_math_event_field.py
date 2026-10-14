#!/usr/bin/env python3
"""HM72: observe event fields, not a more accurate source-chart predictor."""

from __future__ import annotations

import argparse
import itertools
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_event_field as M
from grain_research.construction.math_answer_recoding_mechanism import render
from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from scripts.run_grain_math_construct_routing import save, write
from scripts.run_grain_math_evaluative_routing import load_arrays
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_math_span_binding import hidden_tensor, load_observer
from scripts.run_grain_upstream_routes import collector_guard

OUT = ROOT / "results/grain_math_response_v1/event_field_presentation_r1"
OLD = ROOT / "results/grain_math_response_v1/ownership_composition_r1"
COLD = ROOT / "results/grain_math_response_v1/cold_relational_measurement_r1"
PROFILE = (
    ROOT
    / "results/grain_math_response_v1/author_observation_accounts_r1/01/profiles.npz"
)
PLAN = "docs/GRAIN_EVENT_FIELD_PRESENTATION_PLAN_HM72.md"
CODE = (
    PLAN,
    "grain_research/construction/math_event_field.py",
    "scripts/run_grain_math_event_field.py",
    "tests/test_grain_math_event_field.py",
    "grain_research/construction/math_cold_relational.py",
    "grain_research/construction/math_answer_recoding_mechanism.py",
    "scripts/run_grain_math_span_binding.py",
    "grain_research/vector_mechanisms/trace.py",
)
MAX_CALLS, MAX_SECONDS = 960, 600


def prepare(out):
    """Freeze complete support, actual prefix/spans and existing-state references."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve previous packets; no overwrite")
    if (
        read_json(COLD / "verification.json")["status"]
        != "VERIFIED_FINITE_HM71_NOT_GLOBAL"
    ):
        raise ValueError("Verified terminal HM71 required")
    source, cold = read_json(OLD / "00/rows.json"), read_json(COLD / "00/rows.json")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    system = read_json(ROOT / "configs/grain_math_passive_event_composition_v1.json")[
        "system"
    ]
    refs = {}
    for bank, root, old_rows in (("source", OLD, source), ("new", COLD, cold)):
        for r in old_rows:
            if bank == "source" and r["family"] != 0:
                continue
            if (r["focal"], r["companion"]) not in M.PAIRS:
                continue
            pair = M.PAIRS.index((r["focal"], r["companion"]))
            order = 0 if bank == "source" or r["form"] == "paraphrase" else 1
            refs.setdefault((bank, pair, r["corner"], order), (root, r))
    rows, reference_files = [], {}
    for doc in M.documents(source, cold):
        text, ids, part, _, _ = render(
            doc["body"], "extraversion", "A", 1, system, tokenizer
        )
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        spans = M.positions(text, doc["body"], enc["offset_mapping"])
        endpoint = part["body"][-1]
        if max(p for a in spans.values() for p in a["event"]) != endpoint:
            raise ValueError("Body endpoint must be the last actual event token")
        prefix = ids[: endpoint + 1]
        ref = refs.get((doc["bank"], doc["pair"], doc["cell"], doc["order"]))
        reference = None
        if ref is not None:
            root, r = ref
            if prefix != r["input_ids"][: r["partition"]["body"][-1] + 1]:
                raise ValueError("Changed original body prefix")
            path = root / f"01/units/unit_{r['unit']:04d}.json"
            load_arrays(path)
            reference = str(path.relative_to(ROOT))
            for p in (path, path.with_suffix(".npz")):
                reference_files[str(p.relative_to(ROOT))] = digest(p)
        rows.append(doc | {"input_ids": prefix, "spans": spans, "reference": reference})
    if len(rows) != MAX_CALLS:
        raise ValueError("Complete all30 pairs and both banks/orders required")
    proofs = []
    for a, b in zip(rows[::2], rows[1::2], strict=True):
        if a["body"].split("\n") != b["body"].split("\n")[::-1]:
            raise ValueError("Exact sentence reversal required")
        if Counter(a["input_ids"]) != Counter(b["input_ids"]):
            raise ValueError("Reversal changed actual complete token bag")
        proofs.append(
            {"forward": a["unit"], "reverse": b["unit"], "same_token_bag": True}
        )
    for directory in ("00/code", "01/units", "03"):
        (out / directory).mkdir(parents=True, exist_ok=True)
    write(out / "00/rows.json", rows)
    write(out / "00/order_proofs.json", proofs)
    code = {}
    for name in CODE:
        path = out / "00/code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
        code[name] = digest(path)
    parents = {
        str(p.relative_to(ROOT)): digest(p)
        for p in (
            OLD / "00/rows.json",
            COLD / "00/rows.json",
            PROFILE,
            MODEL / "config.json",
        )
        if p.is_relative_to(ROOT)
    }
    write(
        out / "00/manifest.json",
        {
            "packet": "HM72",
            "code": code,
            "assets": {
                n: digest(out / n) for n in ("00/rows.json", "00/order_proofs.json")
            },
            "parents": parents,
            "references": reference_files,
            "model_config_sha256": digest(MODEL / "config.json"),
            "layer": 16,
            "max_attempts": MAX_CALLS,
            "max_wall_seconds": MAX_SECONDS,
            "no_fit_API_private_training_generation": True,
            "event_equivalence_design_declared": True,
            "original_scope_complete": False,
        },
    )
    write(
        out / "run_status.json",
        {
            "status": "PREPARED",
            "attempts": 0,
            "completed": 0,
            "in_flight": None,
            "wall_seconds": 0,
        },
    )
    print(
        f"HM72 prepared {len(rows)} actual prefix units, {len(proofs)} reorder proofs"
    )


def check(out):
    """Validate live/frozen code and fixed material without changing old results."""
    m = read_json(out / "00/manifest.json")
    for name, sha in m["code"].items():
        if digest(ROOT / name) != sha or digest(out / "00/code" / name) != sha:
            raise ValueError("Frozen code drift: " + name)
    for name, sha in m["assets"].items():
        if digest(out / name) != sha:
            raise ValueError("Material drift: " + name)
    for name, sha in (m["parents"] | m["references"]).items():
        if digest(ROOT / name) != sha:
            raise ValueError("Parent/reference drift: " + name)
    if digest(MODEL / "config.json") != m["model_config_sha256"]:
        raise ValueError("Frozen model config drift")
    return m


class CutReached(Exception):
    """A successful early capture, not an uncertain model failure."""


def capture(observer, row):
    """Capture the causal field once and prohibit all blocks above the frozen cut."""
    import torch

    state, upper = {}, []

    def stop(_module, _args, output):
        state["hidden"] = hidden_tensor(output)[0].detach().cpu().numpy().copy()
        raise CutReached

    def upper_guard(_module, _args):
        upper.append(True)
        raise RuntimeError("No upper block permitted")

    handles = [observer.model.model.layers[16].register_forward_hook(stop)]
    handles.append(
        observer.model.model.layers[17].register_forward_pre_hook(upper_guard)
    )
    tokens = torch.tensor(
        [row["input_ids"]],
        dtype=torch.long,
        device=next(observer.model.parameters()).device,
    )
    try:
        try:
            with torch.inference_mode():
                observer.model.model(
                    input_ids=tokens,
                    attention_mask=torch.ones_like(tokens),
                    use_cache=False,
                    return_dict=True,
                )
        except CutReached:
            pass
        if (
            upper
            or "hidden" not in state
            or len(state["hidden"]) != len(row["input_ids"])
        ):
            raise ValueError("Early capture was incomplete or upper model executed")
        observer.calls += 1
        observer.input_tokens += len(row["input_ids"])
        observer.processed_tokens += len(row["input_ids"])
        return M.extract(state["hidden"], row["spans"])
    finally:
        for handle in handles:
            handle.remove()


def collect(out):
    """Persist each attempt and field, never retry an uncertain interrupted call."""
    check(out)
    status = read_json(out / "run_status.json")
    if status["status"] == "COLLECTED":
        return
    if status["in_flight"] is not None or status["status"] == "STOPPED":
        raise RuntimeError("Uncertain/failed call requires review, not automatic retry")
    if len(list((out / "01/units").glob("unit_*.json"))) != status["completed"]:
        raise ValueError("Status/capture mismatch")
    start, previous = time.monotonic(), status["wall_seconds"]
    try:
        observer = load_observer()
        actual, old = observer.manifest(), read_json(OLD / "01/observer.json")
        for key in (
            "model",
            "torch",
            "transformers",
            "dtype",
            "device",
            "hidden_size",
            "layers",
            "attention",
            "training",
            "template_sha256",
        ):
            if actual[key] != old[key]:
                raise ValueError("Observer drift: " + key)
        write(out / "01/observer.json", actual)
        for row in read_json(out / "00/rows.json"):
            path = out / f"01/units/unit_{row['unit']:04d}.json"
            if path.exists():
                meta, _ = load_arrays(path)
                if any(
                    meta[k] != row[k] for k in ("unit", "bank", "pair", "cell", "order")
                ):
                    raise ValueError("Resume identity mismatch")
                continue
            elapsed = previous + time.monotonic() - start
            if status["attempts"] >= MAX_CALLS or elapsed >= MAX_SECONDS:
                raise TimeoutError("HM72 fixed collector bound")
            status.update(
                status="COLLECTING", in_flight=row["unit"], wall_seconds=elapsed
            )
            status["attempts"] += 1
            write(out / "run_status.json", status)
            arrays = capture(observer, row)
            difference = None
            if row["reference"] is not None:
                _, old_arrays = load_arrays(ROOT / row["reference"])
                difference = float(
                    np.max(abs(arrays["body_end"] - old_arrays["body_end"]))
                )
            save(
                path,
                {k: row[k] for k in ("unit", "bank", "pair", "cell", "order")}
                | {
                    "upper_blocks_called": 0,
                    "existing_endpoint_max_difference": difference,
                },
                arrays,
            )
            status.update(
                completed=status["completed"] + 1,
                in_flight=None,
                wall_seconds=previous + time.monotonic() - start,
            )
            write(out / "run_status.json", status)
            if status["completed"] % 120 == 0:
                print(
                    f"HM72 ACTUAL {status['completed']}/960;{status['wall_seconds']:.1f}s",
                    flush=True,
                )
        if status["completed"] != MAX_CALLS or status["attempts"] != MAX_CALLS:
            raise ValueError("Incomplete acquisition")
        status["status"] = "COLLECTED"
        write(out / "run_status.json", status)
        paths = [out / "run_status.json", out / "01/observer.json"]
        paths.extend(sorted((out / "01/units").glob("unit_*.*")))
        write(
            out / "01/raw_seal.json",
            {str(p.relative_to(out)): digest(p) for p in paths},
        )
    except BaseException as error:
        status.update(
            status="STOPPED",
            error=repr(error),
            wall_seconds=previous + time.monotonic() - start,
        )
        write(out / "run_status.json", status)
        raise


def tables(out):
    """Rebuild all field tables with complete cell/order identities and raw hashes."""
    for name, sha in read_json(out / "01/raw_seal.json").items():
        if digest(out / name) != sha:
            raise ValueError("Raw drift: " + name)
    data, seen, discrepancies = {}, set(), []
    for row in read_json(out / "00/rows.json"):
        meta, arrays = load_arrays(out / f"01/units/unit_{row['unit']:04d}.json")
        identity = tuple(row[k] for k in ("bank", "pair", "order", "cell"))
        if identity in seen or meta["upper_blocks_called"] != 0:
            raise ValueError("Duplicate capture or upper execution")
        seen.add(identity)
        if any(meta[k] != row[k] for k in ("unit", "bank", "pair", "order", "cell")):
            raise ValueError("Raw identity mismatch")
        if meta["existing_endpoint_max_difference"] is not None:
            discrepancies.append(meta["existing_endpoint_max_difference"])
        for feature in M.FEATURES:
            a = np.asarray(arrays[feature], float)
            if not np.isfinite(a).all():
                raise ValueError("Nonfinite field")
            key = row["bank"], row["pair"], feature
            table = data.setdefault(key, np.empty((2, 8, *a.shape)))
            table[row["order"], row["cell"]] = a
    if len(seen) != 960 or len(data) != 300:
        raise ValueError("Incomplete full-scope field support")
    return data, discrepancies


def compute(data):
    """Contract all fixed profiles and characters, keeping all field/order variants."""
    with np.load(PROFILE, allow_pickle=False) as a:
        p = a["probabilities"].reshape(30, 8, 2, 8)
    rows, coefficients, summary, saved = [], [], [], {}
    for pair, feature in itertools.product(range(30), M.FEATURES):
        banks = {b: M.decompose(data[b, pair, feature]) for b in ("source", "new")}
        for mode in M.MODES:
            source, new = banks["source"][mode], banks["new"][mode]
            for case in range(8):
                a, b = M.observe(p[pair, case], source), M.observe(p[pair, case], new)
                wrong = M.observe(p[pair, case], np.roll(new, 1, axis=0))
                rows.append(
                    {"pair": pair, "case": case, "feature": feature, "mode": mode}
                    | M.relation_row(a, b)
                    | {"wrong_cell_cosine": M.cosine(a, wrong)}
                )
                saved[f"{pair:02d}_{feature}_{mode}_case{case}"] = np.stack((a, b))
            for label, character in zip(M.CHARACTERS, M.characters(), strict=True):
                a, b = (
                    np.tensordot(character / 8, source, axes=1),
                    np.tensordot(character / 8, new, axes=1),
                )
                coefficients.append(
                    {"pair": pair, "character": label, "feature": feature, "mode": mode}
                    | M.relation_row(a, b)
                )
    for feature, mode in itertools.product(M.FEATURES, M.MODES):
        selected = [r for r in rows if r["feature"] == feature and r["mode"] == mode]
        result = {"feature": feature, "mode": mode, "profile_rows": len(selected)}
        for field in ("direction_cosine", "wrong_slot_cosine", "wrong_cell_cosine"):
            values = [r[field] for r in selected if r[field] is not None]
            result[field + "_defined"] = len(values)
            result[field + "_mean"] = (
                math.fsum(values) / len(values) if values else None
            )
        for field in ("source_norm", "new_norm", "error_norm"):
            result[field + "_RMS"] = math.sqrt(
                math.fsum(r[field] ** 2 for r in selected) / len(selected)
            )
        summary.append(result)
    return rows, coefficients, summary, saved


def analyze(out):
    """Seal actual observations and finite direction results, not an accuracy ranking."""
    check(out)
    if (out / "03/output_seal.json").exists():
        raise FileExistsError("Do not overwrite analyzed results")
    data, discrepancies = tables(out)
    rows, coefficients, summary, contrasts = compute(data)
    write_csv(out / "03/profile_directions.csv", rows)
    write_csv(out / "03/boolean_coefficients.csv", coefficients)
    np.savez_compressed(
        out / "03/field_tables.npz",
        **{f"{b}_{p:02d}_{f}": a for (b, p, f), a in data.items()},
    )
    np.savez_compressed(out / "03/profile_vectors.npz", **contrasts)
    write(
        out / "03/summary.json",
        {
            "packet": "HM72",
            "attempts": read_json(out / "run_status.json")["attempts"],
            "actual_calls": 960,
            "profile_rows": len(rows),
            "boolean_rows": len(coefficients),
            "wall_seconds": read_json(out / "run_status.json")["wall_seconds"],
            "reference_endpoint_count": len(discrepancies),
            "reference_endpoint_max_difference": max(discrepancies),
            "upper_blocks_executed": 0,
            "summaries": summary,
            "no_fit_API_private_labels_training": True,
            "original_scope_complete": False,
            "invariance_of_even_is_algebra_not_empirical_discovery": True,
        },
    )
    paths = sorted((out / "03").glob("*"))
    write(
        out / "03/output_seal.json", {str(p.relative_to(out)): digest(p) for p in paths}
    )
    print(
        f"HM72 analyzed {len(rows)} profiles, {len(coefficients)} coefficients",
        flush=True,
    )


def verify(out):
    """Recompute all exports and scalar contractions using extended precision."""
    check(out)
    for name, sha in read_json(out / "03/output_seal.json").items():
        if digest(out / name) != sha:
            raise ValueError("Primary result drift")
    data, _ = tables(out)
    rows, coefficients, summary, computed = compute(data)
    with np.load(out / "03/profile_vectors.npz", allow_pickle=False) as a:
        for key, values in computed.items():
            np.testing.assert_array_equal(a[key], values)
    if summary != read_json(out / "03/summary.json")["summaries"]:
        raise ValueError("Summary replay difference")
    with np.load(PROFILE, allow_pickle=False) as a:
        profiles = a["probabilities"].reshape(30, 8, 2, 8)
    maximum, count = 0.0, 0
    for bank, pair, feature in data:
        raw = data[bank, pair, feature]
        modes = M.decompose(raw)
        np.testing.assert_allclose(modes["even"] + modes["odd"], raw[0], atol=1e-12)
        np.testing.assert_allclose(modes["even"] - modes["odd"], raw[1], atol=1e-12)
        np.testing.assert_allclose(
            np.sum(raw**2) / 2,
            np.sum(modes["even"] ** 2) + np.sum(modes["odd"] ** 2),
            rtol=1e-12,
        )
        for mode, field in modes.items():
            weights = [p[1] - p[0] for p in profiles[pair]] + list(M.characters() / 8)
            for index, d in enumerate(weights):
                expanded = np.asarray(d, np.longdouble).reshape(
                    (8,) + (1,) * (field.ndim - 1)
                )
                alternate = np.sum(
                    expanded * np.asarray(field, np.longdouble),
                    axis=0,
                    dtype=np.longdouble,
                ).astype(float)
                regular = np.tensordot(d, field, axes=1)
                np.testing.assert_allclose(regular, alternate, atol=2e-12, rtol=1e-12)
                maximum = max(maximum, float(np.max(abs(regular - alternate))))
                count += regular.size
                if index < 8:
                    item = computed[f"{pair:02d}_{feature}_{mode}_case{index}"][
                        int(bank == "new")
                    ]
                    np.testing.assert_allclose(item, alternate, atol=2e-12, rtol=1e-12)
    if len(rows) != 4800 or len(coefficients) != 4200:
        raise ValueError("Incomplete profile/character battery")
    write(
        out / "verification.json",
        {
            "status": "VERIFIED_FINITE_HM72_NOT_GLOBAL",
            "scalar_checks": count,
            "max_error": maximum,
            "same_agent_alternate_not_third_party": True,
            "original_scope_complete": False,
        },
    )
    write(
        out / "checkpoint.json",
        {
            "status": "VERIFIED_FINITE_HM72_NOT_GLOBAL",
            "active_processes": [],
            "original_scope_complete": False,
        },
    )
    print(
        f"HM72 verified {count} extended-precision scalars,maxerror={maximum:.3g}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "collect", "analyze", "verify"))
    args = parser.parse_args()
    if args.action == "collect":
        with collector_guard(OUT):
            collect(OUT)
    else:
        {"prepare": prepare, "analyze": analyze, "verify": verify}[args.action](OUT)
