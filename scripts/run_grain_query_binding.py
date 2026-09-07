#!/usr/bin/env python3
"""Bounded N/K/R query-region factorial with size-matched controls and cached endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
import zipfile
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.trace import DecoderObserver, Patch
from scripts.run_grain_persona_reciprocity import (
    EXPECTED_METADATA,
    BudgetExceeded,
    file_hash,
)
from scripts.run_grain_vector_mechanism_controls import write_json

MASKS = list(product((0, 1), repeat=3))


def word_intervals(row):
    """Derive only predeclared authored clause operands, not effect-selected tokens."""
    clause = row["clauses"]["C"]
    a, b = row["intervals"]["C"]
    if row["variant"] == "venue":
        name, criterion = "room", "open"
    else:
        match = re.search(r"(?:Did|does) (Alex|Blair)\b", clause)
        if match is None:
            raise ValueError("Authored entity grammar changed")
        name = match.group(1)
        criterion = (
            "initiate the conversation rather than wait"
            if row["variant"] == "event_target"
            else "outgoing rather than reserved"
        )
    result = {}
    for key, value in (("N", name), ("K", criterion)):
        part = row["user"][a:b]
        if part.count(value) != 1:
            raise ValueError("Declared operand not unique in question")
        start = a + part.index(value)
        end = start + len(value)
        if start > a and row["user"][start - 1] == " ":
            start -= 1
        result[key] = {"text": value, "span": [start, end]}
    return result


def matched_random(q, chosen, identity):
    """Sample an equal-count complement with a paired seed independent of outcomes."""
    pool = sorted(set(q) - set(chosen))
    if not chosen or len(pool) < len(chosen):
        raise ValueError("Count-matched complement unavailable")
    seed = int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)
    return sorted(
        np.random.default_rng(seed).choice(pool, len(chosen), replace=False).tolist()
    )


def anchored_terms(values):
    """Return the exact seven binary interaction contrasts, without mediation claims."""
    if set(values) != set(MASKS):
        raise ValueError("Complete factorial required")
    result = {}
    for n in (1, 2, 3):
        for subset in combinations(range(3), n):
            result["".join("NKR"[j] for j in subset)] = sum(
                (-1) ** (n - sum(m)) * value
                for m, value in values.items()
                if all(not m[j] for j in range(3) if j not in subset)
            )
    return result


def analyze(records, native):
    """Keep per-case controls and complete nonadditivity before any aggregation."""
    rows = pd.DataFrame(records)
    lookup = native.set_index("case_id").margin
    rows["shift"] = rows.margin - rows.case_id.map(lookup)
    rows["gap"] = rows.donor_case.map(lookup) - rows.case_id.map(lookup)
    rows["error"] = rows.margin - rows.donor_case.map(lookup)
    summary, terms, controls = [], [], []
    for (variant, form, mask), sub in rows.groupby(["variant", "form", "mask"]):
        energy = float(np.sum(sub.gap**2))
        summary.append(
            {
                "variant": variant,
                "form": form,
                "mask": mask,
                "cases": len(sub),
                "relative_error_energy": float(np.sum(sub.error**2) / energy),
                "shift_rms": float(np.sqrt(np.mean(sub["shift"] ** 2))),
                "gap_rms": float(np.sqrt(energy / len(sub))),
            }
        )
    for case, sub in rows.groupby("case_id"):
        meta = sub.iloc[0]
        sub = sub.set_index("mask")
        values = {m: float(sub.loc["".join(map(str, m)), "shift"]) for m in MASKS}
        for term, value in anchored_terms(values).items():
            terms.append(
                {
                    "case_id": case,
                    "variant": meta.variant,
                    "form": meta.form,
                    "group_id": meta.group_id,
                    "level": int(meta.level),
                    "term": term,
                    "value": value,
                }
            )
        for name, mask in (("N", "100"), ("K", "010")):
            controls.append(
                {
                    "case_id": case,
                    "variant": meta.variant,
                    "form": meta.form,
                    "group_id": meta.group_id,
                    "level": int(meta.level),
                    "region": name,
                    "specific_squared_error": float(sub.loc[mask, "error"] ** 2),
                    "random_squared_error": float(
                        sub.loc["random_" + name, "error"] ** 2
                    ),
                    "specific_shift": float(sub.loc[mask, "shift"]),
                    "random_shift": float(sub.loc["random_" + name, "shift"]),
                    "gap_squared": float(meta.gap**2),
                }
            )
    return {
        "response_rows": rows,
        "transport_summary": pd.DataFrame(summary),
        "interaction_terms": pd.DataFrame(terms),
        "matched_random_controls": pd.DataFrame(controls),
    }


def run(args):
    """Collect only missing factorial cells, preserving parent endpoints and all controls."""
    parent = json.loads((args.parent_dir / "manifest.json").read_text())
    state = json.loads((args.parent_dir / "run_status.json").read_text())
    if state["status"] != "COMPLETE" or state["calls"] != 340:
        raise ValueError("Complete P1H required")
    for name, expected in json.loads(
        (args.parent_dir / "output_hashes.json").read_text()
    ).items():
        if file_hash(args.parent_dir / name) != expected:
            raise ValueError("Parent data changed")
    data = json.loads((args.parent_dir / "materials.json").read_text())
    rows = data["rows"]
    native = pd.read_csv(args.parent_dir / "baselines.csv")
    lookup_margin = native.set_index("case_id").margin
    parent_q = (
        pd.read_csv(args.parent_dir / "effects.csv")
        .query("patch == 'Q'")
        .set_index("case_id")
    )
    layouts = json.loads((args.parent_dir / "token_layouts.json").read_text())
    states = {
        r["case_id"]: np.load(
            args.parent_dir / "activations" / (r["case_id"] + ".npz"),
            allow_pickle=False,
        )["block_16"]
        for r in rows
    }
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/trace.py",
        ROOT / "grain_research/vector_mechanisms/multicomment.py",
        ROOT / "grain_research/vector_mechanisms/controls.py",
        ROOT / "scripts/run_grain_persona_reciprocity.py",
        ROOT / "scripts/run_grain_vector_mechanism_controls.py",
        ROOT / "docs/GRAIN_QUERY_BINDING_P1I_PROTOCOL.md",
        ROOT / "tests/test_grain_query_binding.py",
    ]
    manifest = {
        "parent_identity": parent["identity"],
        "material_hash": parent["material_hash"],
        "max_calls": 500,
        "max_seconds": 600,
        "private_data_read": False,
        "code": {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    with zipfile.ZipFile(out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for path in sources:
            z.write(path, path.relative_to(ROOT))
    observer, start = None, time.monotonic()
    records, checks = [], []
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )

    def flush(status, **extra):
        """Persist all cells with their reused-versus-new source distinguished."""
        pd.DataFrame(records).to_csv(out / "scores.csv", index=False)
        write_json(out / "checks.json", checks)
        status = {
            "status": status,
            "identity": manifest["identity"],
            "calls": observer.calls if observer else 0,
            "seconds": time.monotonic() - start,
            "rows": len(records),
            **extra,
        }
        write_json(out / "run_status.json", status)
        return status

    try:
        expected = EXPECTED_METADATA | {
            "model.safetensors": "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
        }
        if {k: file_hash(args.model_dir / k) for k in expected} != expected:
            raise ValueError("Model changed")
        torch.set_num_threads(2)
        observer = DecoderObserver.load(
            str(args.model_dir), device="mps", dtype="float32", max_tokens=512
        )
        observer.sites = {"block_16": observer.model.model.layers[16]}
        write_json(out / "model_manifest.json", observer.manifest())
        regions = {}
        for r in rows:
            case = r["case_id"]
            layout = layouts[case]
            if observer.prompt_ids(data["system"], r["user"]) != layout["ids"]:
                raise ValueError("Tokenization changed")
            intervals = word_intervals(r)
            parts = {
                key: observer.content_intervals(
                    data["system"], r["user"], [value["span"]]
                )
                for key, value in intervals.items()
            }
            if set(parts["N"]) & set(parts["K"]):
                raise ValueError("Overlapping operands")
            q = layout["partitions"]["Q"]
            parts["R"] = sorted(set(q) - set(parts["N"]) - set(parts["K"]))
            if sorted(j for p in parts.values() for j in p) != q:
                raise ValueError("Question coverage failed")
            pair_id = "__".join(str(r[k]) for k in ("group_id", "variant", "form"))
            controls = {
                "random_" + key: matched_random(q, parts[key], pair_id + key)
                for key in ("N", "K")
            }
            regions[case] = {
                "intervals": intervals,
                "parts": parts,
                "controls": controls,
            }
        for r in rows:
            case = r["case_id"]
            donor = parent_q.loc[case, "donor_case"]
            if regions[case] != regions[donor]:
                raise ValueError("Paired coordinates differ")
        write_json(out / "regions.json", regions)

        def budget(n=1):
            """Bound actual forwards and finish cases only within the declared budget."""
            if (
                observer.calls + n > 500
                or time.monotonic() - start > 600
                or torch.mps.current_allocated_memory() > 9 * 1024**3
            ):
                raise BudgetExceeded("P1I resource cap")

        def score(r, patch=None):
            """Capture native single-token candidates for exact score replay."""
            budget()
            ids = layouts[r["case_id"]]["ids"]
            _, logits = observer.capture(ids, patch=patch)
            hi, lo = [
                observer.candidate_ids(ids, r[k])[0]
                for k in ("high_answer", "low_answer")
            ]
            return (
                float(logits[hi]) - float(logits[lo]),
                float(logits[hi]),
                float(logits[lo]),
            )

        for i, r in enumerate(rows):
            budget(10)
            case = r["case_id"]
            donor = parent_q.loc[case, "donor_case"]
            if i < 2:
                value, _, _ = score(r)
                indices = regions[case]["parts"]["K"]
                same, _, _ = score(r, Patch("block_16", indices, states[case][indices]))
                error = max(abs(value - lookup_margin[case]), abs(same - value))
                checks.append({"case_id": case, "native_self_error": error})
                if error > 1e-5:
                    raise ValueError("Parent native/self replay failed")
            meta = {
                key: r[key]
                for key in ("case_id", "variant", "form", "group_id", "level")
            } | {"donor_case": donor}
            for mask in MASKS:
                name = "".join(map(str, mask))
                if name in ("000", "111"):
                    source = (
                        native.set_index("case_id").loc[case]
                        if name == "000"
                        else parent_q.loc[case]
                    )
                    records.append(
                        meta
                        | {
                            "mask": name,
                            "margin": float(source.margin),
                            "high_logit": float(source.high_logit),
                            "low_logit": float(source.low_logit),
                            "source": "parent",
                        }
                    )
                    continue
                indices = sorted(
                    j
                    for key, bit in zip("NKR", mask)
                    if bit
                    for j in regions[case]["parts"][key]
                )
                value, hi, lo = score(
                    r, Patch("block_16", indices, states[donor][indices])
                )
                records.append(
                    meta
                    | {
                        "mask": name,
                        "margin": value,
                        "high_logit": hi,
                        "low_logit": lo,
                        "source": "new",
                    }
                )
            for name, indices in regions[case]["controls"].items():
                value, hi, lo = score(
                    r, Patch("block_16", indices, states[donor][indices])
                )
                records.append(
                    meta
                    | {
                        "mask": name,
                        "margin": value,
                        "high_logit": hi,
                        "low_logit": lo,
                        "source": "new",
                    }
                )
            if (i + 1) % 12 == 0:
                flush("PATCH_RUNNING")
                print(f"P1I {i + 1}/48 calls={observer.calls}", flush=True)
        if observer.calls != 388 or len(records) != 480:
            raise ValueError("Incomplete packet")
        for name, frame in analyze(records, native).items():
            frame.to_csv(out / (name + ".csv"), index=False)
        result = flush("COMPLETE")
        write_json(
            out / "output_hashes.json",
            {
                p.name: file_hash(p)
                for p in out.iterdir()
                if p.is_file() and not p.name.startswith("._")
            },
        )
        print(json.dumps(result), flush=True)
    except Exception as exc:
        flush(
            "PARTIAL_BUDGET" if isinstance(exc, BudgetExceeded) else "FAILED",
            error=str(exc),
        )
        (out / "error_log.txt").write_text(traceback.format_exc())
        raise


def main():
    """Run the fixed offline donor panel without any endpoint fitting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT
        / "results/grain_vector_mechanisms/local_observers/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
