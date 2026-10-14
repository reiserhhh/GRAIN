#!/usr/bin/env python3
"""HM86: source-calibrated, score-free query/body attention mechanism experiment."""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_query_evidence as M
from grain_research.construction.math_answer_recoding_mechanism import render
from grain_research.construction.math_passive_event_composition import action_line
from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from grain_research.vector_mechanisms.prefix import capture_prefix, rotate_query
from scripts.run_grain_author_contrast_hm82 import table, write
from scripts.run_grain_math_construct_routing import save
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_query_evidence_hm86.json"
OUT = ROOT / "results/grain_math_response_v1/query_evidence_routing_r1"
VIEWS = ("native", "neutral_reference")
READOUTS = (
    "actual",
    "conditional",
    "event_uniform",
    "pairing",
    "conditional_uniform",
    "conditional_pairing",
    "uniform_body",
)
CODE = (
    "configs/grain_query_evidence_hm86.json",
    "docs/GRAIN_QUERY_EVIDENCE_ROUTING_PLAN_HM86.md",
    "grain_research/construction/math_query_evidence.py",
    "scripts/run_grain_query_evidence_hm86.py",
    "scripts/verify_grain_query_evidence_hm86.py",
    "tests/test_grain_math_query_evidence.py",
    "grain_research/vector_mechanisms/prefix.py",
    "grain_research/vector_mechanisms/query_tilt.py",
    "grain_research/construction/math_answer_recoding_mechanism.py",
    "grain_research/construction/math_construct_routing.py",
    "grain_research/construction/math_relational_operation_specificity.py",
    "grain_research/construction/math_passive_event_composition.py",
    "grain_research/vector_mechanisms/trace.py",
    "grain_research/vector_mechanisms/navigation.py",
    "scripts/run_grain_math_span_binding.py",
    "scripts/run_grain_author_contrast_hm82.py",
    "scripts/run_grain_math_construct_routing.py",
    "scripts/run_grain_upstream_routes.py",
)


def check_hashes(base, hashes):
    """Authenticate declared model, code, input or output assets."""
    for name, expected in hashes.items():
        if digest(base / name) != expected:
            raise ValueError("Hash drift: " + name)


def seal(base, folder):
    """Create an explicit ledger of files already written in this folder."""
    return {
        str(p.relative_to(base)): digest(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def tokenize(body, target, system, tokenizer, cut=False):
    """Bind disjoint event token groups to the exact rendered input, not filenames."""
    text, ids, part, _, _ = render(body, target, "A", 1, system, tokenizer)
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)[
        "offset_mapping"
    ]
    start, cursor, groups = text.index(body), 0, []
    for line in body.splitlines():
        lo, hi = start + cursor, start + cursor + len(line)
        group = [i for i, (a, b) in enumerate(offsets) if a < hi and b > lo and b > a]
        groups.append(group)
        cursor += len(line) + 1
    if cut:
        length = part["body"][-1] + 1
        ids, text = ids[:length], text[: offsets[length - 1][1]]
    flat = [i for group in groups for i in group]
    if len(set(flat)) != len(flat) or any(i >= len(ids) for i in flat):
        raise ValueError("Overlapping or truncated event tokens")
    return {"body": body, "rendered": text, "input_ids": ids, "groups": groups}


def prepare(out):
    """Seal all new material and the fixed experiment before collecting vectors."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve old packets and partial failures")
    spec = read_json(SPEC)
    material = read_json(ROOT / spec["source_material"])
    targets = spec["targets"]
    if targets != material["targets"]:
        raise ValueError("Source target roster drift")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    rows = []

    def append(meta, lines, target, cut=False):
        rows.append(
            {
                "unit": len(rows),
                **meta,
                **tokenize(
                    "\n".join(lines), target, material["system"], tokenizer, cut
                ),
            }
        )

    for target, polarity, actor in itertools.product(targets, (-1, 1), ("A", "B")):
        append(
            {"stage": "source", "target": target, "polarity": polarity, "actor": actor},
            [
                action_line(
                    "K1", actor, material["behaviours"][target][0][int(polarity > 0)]
                ),
                action_line(
                    "K2", "B" if actor == "A" else "A", "stood next to a table"
                ),
                action_line("K3", actor, "opened a plain notebook"),
            ],
            target,
            True,
        )
    neutral = [
        action_line("K1", "A", "opened a plain notebook"),
        action_line("K2", "B", "stood next to a table"),
        action_line("K3", "A", "looked at the clock"),
    ]
    for target in targets:
        append({"stage": "reference", "target": target}, neutral, target)
    for context, focal in enumerate(targets):
        companion = targets[(context + 1) % len(targets)]
        for corner, (x, y, g, h) in enumerate(M.points().astype(int).tolist()):
            lines = [
                action_line("K1", "A", spec["fresh_behaviours"][focal][int(x > 0)]),
                action_line(
                    "K2",
                    "A" if g > 0 else "B",
                    spec["fresh_behaviours"][companion][int(y > 0)],
                ),
                action_line("K3", "A" if h > 0 else "B", "opened a plain notebook"),
            ]
            previous = None
            for query in targets:
                append(
                    {
                        "stage": "application",
                        "context": context,
                        "focal": focal,
                        "companion": companion,
                        "corner": corner,
                        "coordinates": [x, y, g, h],
                        "query": query,
                    },
                    lines,
                    query,
                )
                current = rows[-1]["input_ids"][: max(rows[-1]["groups"][-1]) + 1]
                if previous is not None and previous != current:
                    raise ValueError("Query input changed the preceding body tokens")
                previous = current
    if len(rows) != spec["max_forwards"]:
        raise ValueError("Incomplete fixed design")
    model = {
        str(p.relative_to(MODEL)): digest(p)
        for p in sorted(MODEL.iterdir())
        if p.is_file() and p.suffix in (".json", ".safetensors")
    }
    code = {name: digest(ROOT / name) for name in CODE}
    source = {spec["source_material"]: digest(ROOT / spec["source_material"])}
    (out / "00").mkdir(parents=True)
    write(out / "00/rows.json", rows)
    write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "source": source,
            "code": code,
            "model": model,
            "model_path": str(MODEL),
            "input_sha256": digest(out / "00/rows.json"),
            "views": VIEWS,
            "readouts": READOUTS,
            "no_existing_scores": True,
            "mathematics": "standard finite attention/permutation and factorial identities;no transferred PDE theorem",
        },
    )
    print("HM86 prepared", len(rows), "score-free captures", flush=True)


def unit(out, index):
    """Load authenticated captured QKV arrays; never consume answer logits."""
    path = out / f"01/units/unit_{index:04d}.json"
    meta = read_json(path)
    if digest(path.with_suffix(".npz")) != meta["array_sha256"]:
        raise ValueError("Raw unit hash drift")
    if meta["trace"] != {"completed_blocks": list(range(17)), "forbidden_calls": []}:
        raise ValueError("Forbidden upper/scoring execution")
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}


def calibrate(out, rows):
    """Freeze query-free behaviour directions and neutral query seeds before apps."""
    path = out / "01/calibration.json"
    if path.exists():
        unit_meta = read_json(path)
        if digest(path.with_suffix(".npz")) != unit_meta["array_sha256"]:
            raise ValueError("Calibration drift")
        return
    targets = read_json(out / "00/manifest.json")["spec"]["targets"]
    source = []
    for target in targets:
        polarities = []
        for polarity in (-1, 1):
            actors = []
            for actor in ("A", "B"):
                row = next(
                    r
                    for r in rows
                    if r["stage"] == "source"
                    and (r["target"], r["polarity"], r["actor"])
                    == (target, polarity, actor)
                )
                raw = unit(out, row["unit"])
                actors.append(
                    M.event_means(raw["v"], row["groups"], len(raw["q"]))[:, 0]
                )
            polarities.append(actors)
        source.append(polarities)
    d, norms = M.directions(source)
    refs = [
        unit(
            out,
            next(
                r["unit"]
                for r in rows
                if r["stage"] == "reference" and r["target"] == target
            ),
        )["raw_query"]
        for target in targets
    ]
    save(
        path,
        {
            "targets": targets,
            "source_units": 24,
            "reference_units": 6,
            "before_application": True,
            "no_score_or_application_calibration": True,
        },
        {
            "directions": d,
            "source_norms": norms,
            "raw_queries": np.asarray(refs),
            "source_means": np.asarray(source),
        },
    )


def collect(out):
    """Execute a bounded resumable prefix collector, never the model answer head."""
    manifest, rows = (
        read_json(out / "00/manifest.json"),
        read_json(out / "00/rows.json"),
    )
    check_hashes(ROOT, {**manifest["code"], **manifest["source"]})
    check_hashes(MODEL, manifest["model"])
    if digest(out / "00/rows.json") != manifest["input_sha256"]:
        raise ValueError("Input design drift")
    status_path = out / "01/status.json"
    status = (
        read_json(status_path)
        if status_path.exists()
        else {"completed": 0, "model_seconds": 0.0}
    )
    spec, observer = manifest["spec"], load_observer()
    from grain_research.vector_mechanisms.navigation import write_json

    with collector_guard(out):
        try:
            for row in rows:
                if row["stage"] == "application":
                    calibrate(out, rows)
                path = out / f"01/units/unit_{row['unit']:04d}.json"
                if path.exists():
                    unit(out, row["unit"])
                    continue
                if status["model_seconds"] >= spec["max_model_seconds"]:
                    raise TimeoutError("Finite model-time limit; keep checkpoint")
                begin = time.monotonic()
                cache, trace = capture_prefix(
                    observer,
                    row["input_ids"],
                    block=spec["block"],
                    layer=spec["qkv_layer"],
                )
                seconds = time.monotonic() - begin
                cache.pop("boundary")
                save(
                    path,
                    {
                        "unit": row["unit"],
                        "trace": trace,
                        "seconds": seconds,
                        "input_ids_sha256": digest(out / "00/rows.json"),
                        "stage": row["stage"],
                    },
                    cache,
                )
                status = {
                    "completed": row["unit"] + 1,
                    "model_seconds": status["model_seconds"] + seconds,
                    "state": "ACTIVE",
                    "last_unit": row["unit"],
                    "new_calls_this_invocation": observer.calls,
                }
                status_path.parent.mkdir(parents=True, exist_ok=True)
                write_json(status_path, status)
                if status["completed"] % 24 == 0:
                    print(
                        "HM86",
                        status["completed"],
                        "/",
                        len(rows),
                        round(status["model_seconds"], 2),
                        "model seconds",
                        flush=True,
                    )
            status["state"] = "CAPTURES_COMPLETE"
            write_json(status_path, status)
            write(
                out / "01/raw_seal.json",
                seal(out, out / "01/units")
                | {
                    "01/calibration.json": digest(out / "01/calibration.json"),
                    "01/calibration.npz": digest(out / "01/calibration.npz"),
                },
            )
        except BaseException as exc:
            status["state"], status["error"] = "STOPPED_CHECKPOINT", repr(exc)
            status_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(status_path, status)
            raise
    print("HM86 captures complete", status, flush=True)


def analyze(out):
    """Keep all query/role/readout contrasts and reconstruct exact local actions."""
    if (out / "02").exists():
        raise FileExistsError("Preserve analysis and failures")
    manifest, rows = (
        read_json(out / "00/manifest.json"),
        read_json(out / "00/rows.json"),
    )
    check_hashes(ROOT, manifest["code"])
    check_hashes(out, read_json(out / "01/raw_seal.json"))
    targets = manifest["spec"]["targets"]
    with np.load(out / "01/calibration.npz", allow_pickle=False) as arrays:
        dirs, refs = arrays["directions"].copy(), arrays["raw_queries"].copy()
    vectors = np.zeros((6, 16, 6, 2, len(READOUTS), *dirs.shape[1:]))
    mass = np.zeros((6, 16, 6, 2, dirs.shape[1], 3))
    causal, identity, action_error, scalar_count = 0.0, 0.0, 0.0, 0
    reads, first, actions = {}, {}, []
    for row in (r for r in rows if r["stage"] == "application"):
        c, corner, q = row["context"], row["corner"], targets.index(row["query"])
        raw = unit(out, row["unit"])
        end = max(row["groups"][-1]) + 1
        key = c, corner
        if key in first:
            causal = max(
                causal,
                *(
                    float(np.max(np.abs(raw[name][:end] - first[key][name])))
                    for name in ("k", "v")
                ),
            )
        else:
            first[key] = {name: raw[name][:end] for name in ("k", "v")}
        queries = (
            raw["q"],
            rotate_query(
                refs[q], raw["last_cos"], raw["last_sin"], float(raw["scaling"])
            ),
        )
        for view, query in enumerate(queries):
            r = M.reading(raw["k"], raw["v"], query, row["groups"])
            reads[c, corner, q, view] = r
            mass[c, corner, q, view] = r["masses"]
            for channel, name in enumerate(READOUTS):
                vectors[c, corner, q, view, channel] = r[name]
            identity = max(
                identity,
                float(np.max(np.abs(r["actual"] - r["event_uniform"] - r["pairing"]))),
                float(
                    np.max(
                        np.abs(
                            r["conditional"]
                            - r["conditional_uniform"]
                            - r["conditional_pairing"]
                        )
                    )
                ),
            )
            scalar_count += r["actual"].size * 2
    if causal > manifest["spec"]["body_causal_tolerance"]:
        raise ValueError("Future query changed preceding body K/V: " + str(causal))
    scalar = np.einsum("icqvrhd,thd->icqvrt", vectors, dirs)
    summaries, author = [], []
    cube = M.points()
    pplus, pminus = M.evidence_laws(manifest["spec"]["epsilon"])
    for c, q, view in itertools.product(range(6), range(6), range(2)):
        for variable in (0, 1):
            for low in np.flatnonzero(cube[:, variable] < 0):
                partner = cube[low].copy()
                partner[variable] = 1
                high = int(np.flatnonzero(np.all(cube == partner, axis=1))[0])
                a = M.finite_action(reads[c, high, q, view], reads[c, low, q, view])
                action_error = max(
                    action_error,
                    float(
                        np.max(
                            np.abs(
                                a["actual"]
                                - a["value"]
                                - a["allocation"]
                                - a["pairing"]
                            )
                        )
                    ),
                )
                scalar_count += a["actual"].size
                for target, name in enumerate(targets):
                    actions.append(
                        {
                            "context": c,
                            "query": targets[q],
                            "view": VIEWS[view],
                            "variable": "x" if variable == 0 else "y",
                            "low_corner": int(low),
                            "high_corner": high,
                            "g": int(cube[low, 2]),
                            "h": int(cube[low, 3]),
                            "direction": name,
                            **{
                                part: M.project(v, dirs[target])
                                for part, v in a.items()
                            },
                        }
                    )
    for c, view, channel in itertools.product(range(6), range(2), range(len(READOUTS))):
        target = (c + 1) % 6
        s = scalar[c, :, :, view, channel, target]
        own_other = []
        for g in (-1, 1):
            ix = cube[:, 2] == g
            effect = np.mean(s[ix] * cube[ix, 1, None], axis=0) * 2
            matched = effect[target]
            other = np.mean(np.delete(effect, target))
            own_other.append(float(matched))
            summaries.append(
                {
                    "context": c,
                    "target": targets[target],
                    "view": VIEWS[view],
                    "readout": READOUTS[channel],
                    "actor": "A" if g > 0 else "B",
                    "matched_action": float(matched),
                    "other_query_mean": float(other),
                    "selectivity": float(matched - other),
                    "body_mass_mean": float(mass[c, :, :, view].sum(axis=-1).mean()),
                }
            )
        for q, direction in itertools.product(range(6), range(6)):
            values = scalar[c, :, q, view, channel, direction]
            contrast = float((pplus - pminus) @ values)
            coefficient = float((cube[:, 1] * cube[:, 2]) @ values / 16)
            identity = max(
                identity, abs(contrast - 2 * manifest["spec"]["epsilon"] * coefficient)
            )
            ix = cube[:, 3] == -cube[:, 2]
            equal_count = float(
                np.mean(values[ix] * cube[ix, 1] * cube[ix, 2])
                * 2
                * manifest["spec"]["epsilon"]
            )
            author.append(
                {
                    "context": c,
                    "query": targets[q],
                    "view": VIEWS[view],
                    "readout": READOUTS[channel],
                    "direction": targets[direction],
                    "law_contrast": contrast,
                    "yg_coefficient": coefficient,
                    "equal_name_count_law_contrast": equal_count,
                }
            )
    (out / "02").mkdir()
    save(
        out / "02/observations.json",
        {
            "dimensions": "context,corner,query,view,readout,head,dimension",
            "no_target_answer_scores": True,
        },
        {"vectors": vectors, "masses": mass, "projections": scalar},
    )
    table(out / "02/query_selectivity.csv", summaries)
    table(out / "02/finite_actions.csv", actions)
    table(out / "02/author_composition.csv", author)
    compact = []
    for view, name in itertools.product(VIEWS, READOUTS):
        a = [
            r
            for r in summaries
            if r["view"] == view and r["readout"] == name and r["actor"] == "A"
        ]
        b = [
            r
            for r in summaries
            if r["view"] == view and r["readout"] == name and r["actor"] == "B"
        ]
        compact.append(
            {
                "view": view,
                "readout": name,
                "own_positive_selectivity_of6": sum(r["selectivity"] > 0 for r in a),
                "own_selectivity_mean": float(np.mean([r["selectivity"] for r in a])),
                "own_action_mean": float(np.mean([r["matched_action"] for r in a])),
                "foreign_action_mean": float(np.mean([r["matched_action"] for r in b])),
                "per_target_selectivity": {r["target"]: r["selectivity"] for r in a},
            }
        )
    summary = {
        "status": "ANALYZED_NOT_YET_VERIFIED_HM86",
        "primary_question": "pre-answer vector target selection and role-sensitive composition,not prediction accuracy",
        "new_captures": len(rows),
        "source_query_free_captures": 24,
        "fresh_bodies": 96,
        "application_captures": 576,
        "upper_layer_answer_head_API_private_training_calls": [0, 0, 0, 0, 0],
        "fixed_boundary": [16, 17],
        "all_heads_retained": int(dirs.shape[1]),
        "model_seconds": read_json(out / "01/status.json")["model_seconds"],
        "body_query_causality_max_error": causal,
        "algebra_identity_max_error": identity,
        "variable_length_action_max_error": action_error,
        "scalar_identity_checks": scalar_count,
        "conditions": compact,
        "not_psychological_validation": True,
        "source_calibration_not_new_mathematics": True,
        "original_scope_complete": False,
    }
    write(out / "02/summary.json", summary)
    write(out / "02/output_seal.json", seal(out, out / "02"))
    print("HM86 analysis", summary, flush=True)


def main():
    """Run exactly the selected phase; existing artifacts are not replaced."""
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "collect", "analyze", "all"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    for phase in (
        ("prepare", "collect", "analyze") if args.phase == "all" else (args.phase,)
    ):
        globals()[phase](args.out)


if __name__ == "__main__":
    main()
