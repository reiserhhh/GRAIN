#!/usr/bin/env python3
"""Construct from single-event carriers, then examine new qualified author laws."""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_event_context_routes as R
from grain_research.construction import math_qualified_composition as M
from grain_research.construction import math_recursive_collector as C
from grain_research.vector_mechanisms.navigation import MODEL, write_json
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts import run_grain_event_context_routes_hm111 as PREV
from scripts import run_grain_qualified_composition_hm110 as Q
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_recursive_collector_hm112.json"
CODE = tuple(
    dict.fromkeys(
        (
            *PREV.CODE,
            "configs/grain_recursive_collector_hm112.json",
            "docs/GRAIN_RECURSIVE_COLLECTOR_COMPILATION_PLAN_HM112.md",
            "grain_research/construction/math_recursive_collector.py",
            "scripts/run_grain_recursive_collector_hm112.py",
            "scripts/score_grain_recursive_collector_cold_hm112.py",
            "tests/test_grain_math_recursive_collector.py",
        )
    )
)


def check(out):
    """Check frozen code, input rules and actual public pretrained weights."""
    manifest = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, manifest["code"] | manifest["source"])
    H.verify_hashes(out, manifest["inputs"])
    for path, digest in manifest["weights"].items():
        if C.stream_sha(MODEL / path) != digest:
            raise ValueError("Frozen public weights changed")
    return manifest


def prepare(out):
    """Freeze single donors, new triple conditions and the operator before capture."""
    if out.exists():
        raise FileExistsError("Never overwrite research history")
    spec = H.load(SPEC)
    previous = ROOT / spec["previous"]
    PREV.check(previous)
    if (
        H.load(previous / "verification.json")["status"]
        != "VERIFIED_FINITE_HM111_NOT_GLOBAL"
    ):
        raise ValueError("Verified frozen HM111 required")
    old = [row for row in H.load(previous / "00/rows.json") if row["arm"] == [0, 0]]
    reader = H.load(previous / "00/reader.json")
    if reader["targets"] != spec["targets"]:
        raise ValueError("All six source targets and their order required")
    capture = [
        row
        for row in old
        if row["kind"] == "base" or (row["kind"] == "single" and row["owner"] == 1)
    ]
    cases = [
        {"case": i, **{k: row[k] for k in ("bank", "kind", "levels", "owner")}}
        for i, row in enumerate(old)
    ]
    lookup = {(r["bank"], M.key(r)): r for r in old}
    signs, _ = C.parity_laws()
    references = []
    widths = H.load(previous / "00/manifest.json")["slot_widths"]
    starts = [min(e) for e in old[0]["events"]]
    for bank, triple_index, sign, owner in itertools.product(
        range(2), range(len(spec["triples"])), signs, (-1, 1)
    ):
        levels = [1] * 6
        for j, value in zip(spec["triples"][triple_index], sign, strict=True):
            levels[j] = int(value + 1)
        base = lookup[bank, ((1,) * 6, owner)]
        row = {
            **base,
            "kind": "triple",
            "levels": levels,
            "triple_index": triple_index,
            "signs": sign.tolist(),
            "unit": len(references),
            "case": len(cases),
        }
        row["input_ids"] = list(base["input_ids"])
        row["events"] = [list(e) for e in base["events"]]
        for j in spec["triples"][triple_index]:
            singleton = [1] * 6
            singleton[j] = levels[j]
            donor = lookup[bank, (tuple(singleton), 1)]
            row["input_ids"][starts[j] : starts[j] + widths[j]] = donor["input_ids"][
                starts[j] : starts[j] + widths[j]
            ]
            row["events"][j] = list(donor["events"][j])
        occupied = set(
            row["prefix"]
            + row["relay"]
            + row["current"]
            + list(itertools.chain.from_iterable(row["events"]))
        )
        row["padding"] = sorted(set(range(len(row["input_ids"]))) - occupied)
        R.permission_mask(row, 0, 0)
        references.append(row)
        cases.append(
            {
                k: row[k]
                for k in (
                    "case",
                    "bank",
                    "kind",
                    "levels",
                    "owner",
                    "triple_index",
                    "signs",
                )
            }
        )
    if (
        len(capture) != spec["max_library_forwards"]
        or len(references) != spec["max_reference_forwards"]
        or len(cases) != 420
    ):
        raise ValueError("Finite declared support changed")
    for directory in ("00", "01/library_units", "02", "03/reference_units", "04"):
        (out / directory).mkdir(parents=True)
    H.write(out / "00/capture_rows.json", capture)
    H.write(out / "00/reference_rows.json", references)
    H.write(out / "00/cases.json", cases)
    H.write(out / "00/reader.json", reader)
    H.write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "code": {name: H.sha(ROOT / name) for name in CODE},
            "source": {
                str((previous / name).relative_to(ROOT)): H.sha(previous / name)
                for name in (
                    "00/manifest.json",
                    "00/rows.json",
                    "00/reader.json",
                    "verification.json",
                )
            },
            "weights": {
                name: C.stream_sha(MODEL / name)
                for name in ("config.json", "tokenizer.json", "model.safetensors")
            },
            "inputs": H.seals(out, out / "00"),
            "constructor_input_kind": "single/base layer context plus fixed pretrained weights; no joint states/scores",
            "source_or_application_fit": False,
            "new_reference_captured_after_constructor_seal": True,
        },
    )
    for path in ("01/status.json", "03/status.json"):
        write_json(
            out / path,
            {
                "state": "PREPARED",
                "completed": 0,
                "attempts": 0,
                "pending": None,
                "model_seconds": 0.0,
            },
        )
    print("HM112 PREPARED", len(capture), len(references), len(cases), flush=True)


def capture_layers(observer, row):
    """Keep only independently encoded context and constant neutral current seed."""
    layers, saved, handles = observer.model.model.layers, {}, []
    length, dim = len(row["input_ids"]), layers[0].self_attn.head_dim
    end = min(row["current"])

    def projection(layer, name):
        def remember(_module, _args, output):
            saved[layer, name] = (
                output[0, :end]
                .detach()
                .float()
                .cpu()
                .numpy()
                .copy()
                .reshape(end, -1, dim)
            )

        return remember

    def position(_module, _args, kwargs):
        cos, sin = kwargs["position_embeddings"]
        saved["cos"] = cos[0].detach().float().cpu().numpy().copy()
        saved["sin"] = sin[0].detach().float().cpu().numpy().copy()

    def embedding(_module, _args, output):
        saved["seed"] = output[0, row["current"]].detach().float().cpu().numpy().copy()

    try:
        handles.append(
            observer.model.model.embed_tokens.register_forward_hook(embedding)
        )
        handles.append(
            layers[0].self_attn.register_forward_pre_hook(position, with_kwargs=True)
        )
        for layer in range(17):
            for name in ("k", "v"):
                handles.append(
                    getattr(
                        layers[layer].self_attn, name + "_proj"
                    ).register_forward_hook(projection(layer, name))
                )
        _discarded_current, trace = R.capture(observer, row)
    finally:
        for handle in handles:
            handle.remove()
    if saved["cos"].shape[0] != length or trace["masked_blocks"] != list(range(17)):
        raise RuntimeError("Incomplete context capture")
    return {
        name: np.stack([saved[layer, name] for layer in range(17)])
        for name in ("k", "v")
    } | {name: saved[name] for name in ("cos", "sin", "seed")}, trace


def collect(out, phase):
    """Commit actual finite calls; never repeat a pending uncertain forward."""
    manifest = check(out)
    spec = manifest["spec"]
    library = phase == "libraries"
    directory, rows_name, cap = (
        ("01", "capture_rows", "max_library_seconds")
        if library
        else ("03", "reference_rows", "max_reference_seconds")
    )
    if not library:
        H.verify_hashes(out, H.load(out / "02/construction_seal.json"))
    status_path = out / directory / "status.json"
    status = H.load(status_path)
    rows = H.load(out / f"00/{rows_name}.json")
    if status["pending"] is not None or status["state"] not in (
        "PREPARED",
        "COLLECTING",
    ):
        raise ValueError("Terminal or uncertain capture is not automatically retried")
    observer = load_observer()
    status["state"] = "COLLECTING"
    for index in range(status["completed"], len(rows)):
        if status["model_seconds"] > spec[cap]:
            status["state"] = "BUDGET_STOP"
            write_json(status_path, status)
            raise TimeoutError("Fixed model budget exhausted; preserve checkpoint")
        row = rows[index]
        status["pending"], status["attempts"] = index, status["attempts"] + 1
        write_json(status_path, status)
        started = time.perf_counter()
        try:
            if library:
                arrays, trace = capture_layers(observer, row)
            else:
                raw, trace = R.capture(observer, row)
                arrays = {name: raw[name][row["current"]] for name in ("k", "v")}
            duration = time.perf_counter() - started
            folder = "library_units" if library else "reference_units"
            path = out / directory / folder / f"unit_{index:04d}.npz"
            Q.save(path, **arrays)
            H.write(
                path.with_suffix(".json"),
                {
                    "index": index,
                    "original_unit": row["unit"],
                    "bank": row["bank"],
                    "kind": row["kind"],
                    "levels": row["levels"],
                    "owner": row["owner"],
                    "trace": trace,
                    "seconds": duration,
                    "sha256": H.sha(path),
                    "stored_current_state": not library,
                },
            )
            status["model_seconds"] += duration
            status["completed"], status["pending"] = index + 1, None
            write_json(status_path, status)
            print("HM112", phase, index + 1, "/", len(rows), flush=True)
        except Exception as error:
            status["state"], status["error"] = "FAILED_NOT_RETRIED", repr(error)
            write_json(status_path, status)
            raise
    status["state"] = "COLLECTED"
    write_json(status_path, status)
    H.write(out / directory / "raw_seal.json", H.seals(out, out / directory))


def compile_bank(out, bank):
    """Extract invariance-checked independent donors, then invoke a cold process."""
    manifest = check(out)
    if H.load(out / "01/status.json")["state"] != "COLLECTED":
        raise ValueError("All source/single captures required")
    H.verify_hashes(out, H.load(out / "01/raw_seal.json"))
    rows = H.load(out / "00/capture_rows.json")
    library, checks = {}, []
    for index, row in enumerate(rows):
        if row["bank"] != bank:
            continue
        with np.load(
            out / f"01/library_units/unit_{index:04d}.npz", allow_pickle=False
        ) as z:
            raw = {name: z[name] for name in z.files}
        for name in ("cos", "sin", "seed"):
            if name in library and not np.array_equal(library[name], raw[name]):
                raise ValueError("Current seed/position is not common")
            library.setdefault(name, raw[name])
        names = [("p", row["prefix"]), (f"r_o{row['owner']}", row["relay"])] + [
            (f"e{j}_l{level}", row["events"][j])
            for j, level in enumerate(row["levels"])
        ]
        for name, positions in names:
            pos = np.array(positions, int)
            if name + "_pos" in library and not np.array_equal(
                pos, library[name + "_pos"]
            ):
                raise ValueError("Event coordinates changed")
            library.setdefault(name + "_pos", pos)
            for channel in ("k", "v"):
                key, value = name + "_" + channel, raw[channel][:, pos]
                if key in library:
                    error = float(np.max(abs(value - library[key])))
                    checks.append(
                        {
                            "input": index,
                            "bank": bank,
                            "segment": name,
                            "channel": channel,
                            "max_error_all17layers": error,
                        }
                    )
                    if error != 0:
                        raise ValueError("Actual event isolation failed")
                else:
                    library[key] = value
    if len([k for k in library if k.startswith("e") and k.endswith("_k")]) != 18:
        raise ValueError("All six independent events at all three levels required")
    path = out / f"02/bank_{bank}_library.npz"
    Q.save(path, **library)
    H.write(out / f"02/bank_{bank}_isolation.json", checks)
    reader = H.load(out / "00/reader.json")
    cases = [r for r in H.load(out / "00/cases.json") if r["bank"] == bank]
    cold_input = out / f"02/bank_{bank}_input.json"
    H.write(
        cold_input,
        {
            "bank": bank,
            "cases": cases,
            "library": str(path.resolve()),
            "weights": str(MODEL / "model.safetensors"),
            "config": H.load(MODEL / "config.json"),
            "queries": reader["queries"],
            "directions": reader["directions"],
            **{
                key: manifest["spec"][key]
                for key in ("blocks", "variants", "batch_size", "device")
            },
            "max_seconds": manifest["spec"]["max_compiler_seconds_per_bank"],
            "hashes": {
                str(p.resolve()): C.stream_sha(p)
                for p in (
                    path,
                    MODEL / "model.safetensors",
                    ROOT / "grain_research/construction/math_recursive_collector.py",
                    ROOT / "grain_research/construction/math_qualified_composition.py",
                    ROOT / "scripts/score_grain_recursive_collector_cold_hm112.py",
                )
            },
        },
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_recursive_collector_cold_hm112.py"),
            "--input",
            str(cold_input),
            "--output",
            str(out / f"02/bank_{bank}_output.npz"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    H.write(
        out / f"02/bank_{bank}_process.json",
        {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
    )
    if result.returncode:
        raise RuntimeError("Cold construction failed; preserve run and do not tune")
    print("HM112 COMPILED bank", bank, flush=True)


def compile_all(out):
    """Seal every numerical construction before any joint reference is read."""
    for bank in range(2):
        compile_bank(out, bank)
    H.write(out / "02/construction_seal.json", H.seals(out, out / "02"))


def analyze(out):
    """Compare sealed construction and compute all signed qualified author laws."""
    manifest = check(out)
    H.verify_hashes(out, H.load(out / "02/construction_seal.json"))
    H.verify_hashes(out, H.load(out / "03/raw_seal.json"))
    if H.load(out / "03/status.json")["state"] != "COLLECTED":
        raise ValueError("Complete finite triple reference required")
    spec = manifest["spec"]
    previous = ROOT / spec["previous"]
    H.verify_hashes(previous, H.load(previous / "01/raw_seal.json"))
    old = [r for r in H.load(previous / "00/rows.json") if r["arm"] == [0, 0]]
    cases = H.load(out / "00/cases.json")
    reader = H.load(out / "00/reader.json")
    q, d = np.asarray(reader["queries"]), np.asarray(reader["directions"])
    references = []
    for row in old:
        k, v = Q.raw_unit(previous, row)
        references.append((k, v))
    for index in range(spec["max_reference_forwards"]):
        with np.load(
            out / f"03/reference_units/unit_{index:04d}.npz", allow_pickle=False
        ) as z:
            references.append((z["k"], z["v"]))
    reference_scores = np.stack([M.attention(k, v, q, d) for k, v in references])
    reference_k, reference_v = (np.stack([r[i] for r in references]) for i in range(2))
    compiled_k = np.zeros((2, *reference_k.shape), np.float32)
    compiled_v = np.zeros_like(compiled_k)
    compiled_scores = np.zeros((2, len(cases), 6))
    for bank in range(2):
        indices = [r["case"] for r in cases if r["bank"] == bank]
        with np.load(out / f"02/bank_{bank}_output.npz", allow_pickle=False) as z:
            (
                compiled_k[:, indices],
                compiled_v[:, indices],
                compiled_scores[:, indices],
            ) = z["k"], z["v"], z["scores"]
    Q.save(
        out / "04/fields.npz",
        reference_k=reference_k,
        reference_v=reference_v,
        reference_scores=reference_scores,
        compiled_k=compiled_k,
        compiled_v=compiled_v,
        compiled_scores=compiled_scores,
    )
    comparisons, effects, summaries = [], [], []
    for bank, kind, variant in itertools.product(
        range(2), ("legacy", "triple"), range(2)
    ):
        indices = [
            i
            for i, r in enumerate(cases)
            if r["bank"] == bank and (r["kind"] == "triple") == (kind == "triple")
        ]
        max_k = float(np.max(abs(compiled_k[variant, indices] - reference_k[indices])))
        max_v = float(np.max(abs(compiled_v[variant, indices] - reference_v[indices])))
        max_score = float(
            np.max(abs(compiled_scores[variant, indices] - reference_scores[indices]))
        )
        summaries.append(
            {
                "bank": bank,
                "support": kind,
                "variant": spec["variants"][variant],
                "cases": len(indices),
                "key_max_error": max_k,
                "value_max_error": max_v,
                "score_max_error": max_score,
                "score_rms_error": M.rms(
                    compiled_scores[variant, indices] - reference_scores[indices]
                ),
            }
        )
    for i, row in enumerate(cases):
        for variant, target in itertools.product(range(2), range(6)):
            comparisons.append(
                {
                    "case": i,
                    "bank": row["bank"],
                    "kind": row["kind"],
                    "owner": row["owner"],
                    "variant": spec["variants"][variant],
                    "target": spec["targets"][target],
                    "constructed": float(compiled_scores[variant, i, target]),
                    "reference": float(reference_scores[i, target]),
                    "difference": float(
                        compiled_scores[variant, i, target]
                        - reference_scores[i, target]
                    ),
                }
            )
    for bank, triple in itertools.product(range(2), range(4)):
        indices = [
            i
            for i, r in enumerate(cases)
            if r["bank"] == bank
            and r["kind"] == "triple"
            and r["triple_index"] == triple
            and r["owner"] == 1
        ]
        if [cases[i]["signs"] for i in indices] != C.parity_laws()[0].tolist():
            raise ValueError("Qualified triple corners changed")
        for name, field in [
            ("reference", reference_scores),
            *((spec["variants"][v], compiled_scores[v]) for v in range(2)),
        ]:
            scores, coef = C.parity_account(field[indices])
            for t in range(6):
                effects.append(
                    {
                        "bank": bank,
                        "triple": triple,
                        "slots": "+".join(map(str, spec["triples"][triple])),
                        "observer": name,
                        "target": spec["targets"][t],
                        "author_plus": float(scores[0, t]),
                        "author_minus": float(scores[1, t]),
                        "author_difference": float(scores[0, t] - scores[1, t]),
                        "third_coefficient": float(coef[t]),
                        "pair_summary_prediction_difference": 0.0,
                    }
                )
    H.table(out / "04/constructions.csv", comparisons)
    H.table(out / "04/author_three_way.csv", effects)
    H.table(out / "04/numerical_comparisons.csv", summaries)
    counts = C.parity_laws()[1]
    H.write(
        out / "04/summary.json",
        {
            "comparisons": summaries,
            "law_counts": counts.tolist(),
            "author_effect_rms": {
                f"bank{bank}_{name}": M.rms(
                    [
                        r["author_difference"]
                        for r in effects
                        if r["bank"] == bank and r["observer"] == name
                    ]
                )
                for bank, name in itertools.product(
                    range(2), ("reference", *spec["variants"])
                )
            },
            "actual_full_prefix_calls": {
                "library": H.load(out / "01/status.json")["completed"],
                "new_reference": H.load(out / "03/status.json")["completed"],
            },
            "pretrained_neural_dependency": True,
            "fits": 0,
            "not_personality_accuracy": True,
            "constructor_does_not_reconstruct_full11": True,
        },
    )
    H.write(out / "04/output_seal.json", H.seals(out, out / "04"))
    print("HM112 ANALYZED", summaries, flush=True)


def verify(out):
    """Recompute equations, every state/readout and numeric input exclusion."""
    manifest = check(out)
    for phase, name in (
        ("01", "raw_seal"),
        ("02", "construction_seal"),
        ("03", "raw_seal"),
        ("04", "output_seal"),
    ):
        H.verify_hashes(out, H.load(out / f"{phase}/{name}.json"))
    cases = H.load(out / "00/cases.json")
    if any(
        r["kind"] not in ("base", "single")
        for r in H.load(out / "00/capture_rows.json")
    ):
        raise ValueError("Joint source in constructor")
    checks = [r for b in range(2) for r in H.load(out / f"02/bank_{b}_isolation.json")]
    if not checks or any(r["max_error_all17layers"] != 0 for r in checks):
        raise ValueError("Layer-wise event independence failed")
    for bank in range(2):
        cold = H.load(out / f"02/bank_{bank}_input.json")
        receipt = H.load(out / f"02/bank_{bank}_output.json")
        if (
            receipt["transformers_loaded"]
            or receipt["joint_reference_input"]
            or receipt["full_model_forwards"]
            or receipt["cases"] != 210
            or len(cold["hashes"]) != 5
        ):
            raise ValueError("Cold constructor dependencies changed")
    signs, counts = C.parity_laws()
    characters = np.prod(
        signs[:, None, :] ** np.array(list(itertools.product((0, 1), repeat=3)))[None],
        axis=2,
    )
    moments = counts @ characters / 40
    np.testing.assert_allclose(moments[0, :-1], moments[1, :-1], atol=1e-15)
    reader = H.load(out / "00/reader.json")
    q, d = np.asarray(reader["queries"]), np.asarray(reader["directions"])
    maximum_primitive, maximum_law = 0.0, 0.0
    with np.load(out / "04/fields.npz", allow_pickle=False) as z:
        for variant, i in itertools.product(range(2), range(len(cases))):
            actual = M.attention_loop(
                z["compiled_k"][variant, i], z["compiled_v"][variant, i], q, d
            )
            maximum_primitive = max(
                maximum_primitive,
                float(np.max(abs(actual - z["compiled_scores"][variant, i]))),
            )
        for bank, triple, field in itertools.product(
            range(2),
            range(4),
            (z["reference_scores"], z["compiled_scores"][0], z["compiled_scores"][1]),
        ):
            indices = [
                i
                for i, r in enumerate(cases)
                if r["bank"] == bank
                and r["kind"] == "triple"
                and r["triple_index"] == triple
                and r["owner"] == 1
            ]
            values = field[indices]
            score, coef = C.parity_account(values)
            explicit = np.stack([np.repeat(values, w, axis=0).mean(0) for w in counts])
            maximum_law = max(
                maximum_law,
                float(np.max(abs(explicit - score))),
                float(np.max(abs(score[0] - score[1] - 0.8 * coef))),
            )
    summary = H.load(out / "04/summary.json")
    numerical = all(
        max(row["key_max_error"], row["value_max_error"])
        <= manifest["spec"]["state_atol"]
        and row["score_max_error"] <= manifest["spec"]["score_atol"]
        for row in summary["comparisons"]
        if row["variant"] == "layered"
    )
    H.write(
        out / "verification.json",
        {
            "status": "VERIFIED_FINITE_HM112_NOT_GLOBAL"
            if numerical
            else "STRUCTURE_VERIFIED_NUMERICAL_CONSTRUCTION_FAILED",
            "numerical_gate": numerical,
            "layer_isolation_comparisons": len(checks),
            "cold_processes": 2,
            "current_only_arithmetic_executions": 840,
            "primitive_max_error": maximum_primitive,
            "author_law_max_error": maximum_law,
            "original_scope_complete": False,
            "same_agent_alternate_formula_not_third_party": True,
            "no_joint_reference_constructor_inputs": True,
            "full11_reconstruction": False,
        },
    )
    if maximum_primitive > 1e-11 or maximum_law > 1e-11:
        raise ValueError("Numerical replay failed")
    print("HM112 VERIFIED", numerical, maximum_primitive, maximum_law, flush=True)


def main():
    """Run one explicit finite phase; never silently restart a queue or overwrite."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=("prepare", "libraries", "compile", "references", "analyze", "verify"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.phase == "prepare":
        prepare(out)
    elif args.phase in ("libraries", "references"):
        with collector_guard(out):
            collect(out, args.phase)
    else:
        {"compile": compile_all, "analyze": analyze, "verify": verify}[args.phase](out)


if __name__ == "__main__":
    main()
