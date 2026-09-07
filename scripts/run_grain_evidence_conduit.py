#!/usr/bin/env python3
"""Full E/Q/D factorial donor transport on existing synthetic role-swap material."""

from __future__ import annotations

import argparse
import json
import os
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
from grain_research.vector_mechanisms.role_opportunity import (
    SYSTEM,
    build_material,
    check_donor_layout,
    describe,
)
from grain_research.vector_mechanisms.trace import DecoderObserver, Patch
from scripts.run_grain_persona_reciprocity import (
    EXPECTED_METADATA,
    BudgetExceeded,
    file_hash,
)
from scripts.run_grain_vector_mechanism_controls import write_json

SITES = ("block_04", "block_10", "block_16")
MASKS = tuple(product((0, 1), repeat=3))


def conduit_material():
    """Reuse all paired cases in five variants; new36 fact queries share old contexts."""
    core, opportunity = [], []
    for original in build_material():
        r = dict(original)
        if r["family"] == "owner":
            r["variant"] = "event_" + r["owner"]
        elif r["family"] == "trait" and r["question_polarity"] == 1:
            r["variant"] = "trait_code" + str(r["code"])
        elif r["family"] == "venue":
            r["variant"] = "venue"
        elif r["family"] == "opportunity":
            r["variant"] = "opportunity_fact"
            r["user"] = (
                r["evidence"]
                + "\n\nQuestion: Did Alex initiate the conversation rather than wait? Answer Yes or No."
            )
            opportunity.append(r)
            continue
        else:
            continue
        core.append(r)
    if len(core) != 60 or len(opportunity) != 36:
        raise ValueError("Wrong panel size")
    return core, opportunity


def partitions(length: int, span: list[int]) -> dict:
    """Make disjoint exhaustive E/Q/D sets after a shared unchanged prefix."""
    if not span or span[-1] >= length - 1:
        raise ValueError("Evidence reaches decision")
    result = {
        "E": list(span),
        "Q": [i for i in range(span[0], length - 1) if i not in span],
        "D": [length - 1],
    }
    flat = [i for indices in result.values() for i in indices]
    if len(set(flat)) != len(flat) or set(flat) != set(range(span[0], length)):
        raise ValueError("Partitions overlap or omit positions")
    return result


def factorial_interactions(values: dict[tuple, float]) -> dict:
    """Compute anchored inclusion-exclusion contrasts, without a mediation claim."""
    if set(values) != set(MASKS):
        raise ValueError("Full2x2x2 required")
    result = {}
    for size in (1, 2, 3):
        for subset in combinations(range(3), size):
            contrast = 0.0
            for mask in MASKS:
                if any(mask[j] for j in range(3) if j not in subset):
                    continue
                contrast += (-1) ** (size - sum(mask)) * values[mask]
            result["".join("EQD"[j] for j in subset)] = contrast
    return result


def analyze(base, effects, opportunity, previous):
    """Report donor reconstruction and complete factorial interactions per variant."""
    joined = effects.copy()
    lookup = base.set_index("case_id").margin
    joined["baseline"] = joined.case_id.map(lookup)
    joined["donor_baseline"] = joined.donor_case.map(lookup)
    joined["donor_gap"] = joined.donor_baseline - joined.baseline
    joined["residual_error"] = joined.margin - joined.donor_baseline
    joined["shift"] = joined.margin - joined.baseline
    summary, interactions = [], []
    for (variant, site, mask), sub in joined.groupby(["variant", "site", "mask"]):
        denom = float(np.sum(sub.donor_gap**2))
        summary.append(
            {
                "variant": variant,
                "site": site,
                "mask": mask,
                "relative_error_energy": float(np.sum(sub.residual_error**2) / denom)
                if denom > 1e-12
                else None,
                "mean_absolute_error": float(sub.residual_error.abs().mean()),
                "mean_absolute_shift": float(sub["shift"].abs().mean()),
                "donor_gap_rms": float(np.sqrt(np.mean(sub.donor_gap**2))),
                "small_gap_cases": int((sub.donor_gap.abs() < 0.1).sum()),
                "cases": len(sub),
            }
        )
    for (case, site), sub in joined.groupby(["case_id", "site"]):
        vals = {tuple(int(x) for x in r.mask): r.margin for r in sub.itertuples()}
        vals[(0, 0, 0)] = lookup[case]
        for term, value in factorial_interactions(vals).items():
            interactions.append(
                {
                    "case_id": case,
                    "site": site,
                    "variant": sub.variant.iloc[0],
                    "group_id": sub.group_id.iloc[0],
                    "term": term,
                    "value": value,
                }
            )
    tables = {
        "transport_summary": pd.DataFrame(summary),
        "factorial_interactions": pd.DataFrame(interactions),
        "transport_rows": joined,
    }
    previous = previous[previous.family == "opportunity"].copy()
    records = []
    for name, frame in (("fact", opportunity), ("trait", previous)):
        gaps = frame.pivot(
            index=["group_id", "condition"], columns="level", values="margin"
        )
        gaps = (gaps[1] - gaps[-1]).unstack("condition")
        for condition in ("free", "assigned", "neutral"):
            records.append(
                {"query": name, "contrast": condition} | describe(gaps[condition])
            )
        records.append(
            {"query": name, "contrast": "free_minus_assigned"}
            | describe(gaps.free - gaps.assigned)
        )
    tables["opportunity_fact_trait"] = pd.DataFrame(records)
    return tables


def run(args):
    """Collect a fresh finite donor matrix; no adaptation to outcome signs."""
    from scripts.audit_grain_role_opportunity import audit

    receipt = audit(args.previous_dir)
    core, opportunity = conduit_material()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/role_opportunity.py",
        ROOT / "grain_research/vector_mechanisms/trace.py",
        ROOT / "grain_research/vector_mechanisms/analysis.py",
        ROOT / "scripts/run_grain_persona_reciprocity.py",
        ROOT / "scripts/audit_grain_role_opportunity.py",
        ROOT / "scripts/run_grain_vector_mechanism_controls.py",
        ROOT / "grain_research/vector_mechanisms/controls.py",
        ROOT / "grain_research/vector_mechanisms/reciprocity.py",
        ROOT / "docs/GRAIN_EVIDENCE_CONDUIT_P1C_PROTOCOL.md",
        ROOT / "tests/test_grain_evidence_conduit.py",
    ]
    manifest = {
        "parent_identity": receipt["parent_identity"],
        "material_hash": material_hash(core + opportunity),
        "code": {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
        "sites": SITES,
        "max_calls": 1800,
        "max_seconds": 900,
        "private_data_read": False,
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    write_json(
        out / "materials.json",
        {"system": SYSTEM, "core": core, "opportunity": opportunity},
    )
    with zipfile.ZipFile(
        out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sources:
            archive.write(path, path.relative_to(ROOT))
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    start, observer = time.monotonic(), None
    base, effects, native, checks = [], [], [], []
    cache = out / "activations"
    cache.mkdir()

    def flush(status, **extra):
        """Write all complete records and do not disguise budget/failure as completion."""
        for name, data in (
            ("baselines", base),
            ("interventions", effects),
            ("opportunity", native),
        ):
            pd.DataFrame(data).to_csv(out / (name + ".csv"), index=False)
        state = {
            "status": status,
            "identity": manifest["identity"],
            "calls": observer.calls if observer else 0,
            "seconds": time.monotonic() - start,
            "baselines": len(base),
            "interventions": len(effects),
            "opportunity_rows": len(native),
            **extra,
        }
        write_json(out / "run_status.json", state)
        write_json(out / "checks.json", checks)
        return state

    try:
        expected = EXPECTED_METADATA | {
            "model.safetensors": "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
        }
        if {k: file_hash(args.model_dir / k) for k in expected} != expected:
            raise ValueError("Checkpoint changed")
        torch.manual_seed(731)
        torch.set_num_threads(2)
        observer = DecoderObserver.load(
            str(args.model_dir), device="mps", dtype="float32", max_tokens=512
        )
        observer.sites = {
            s: observer.model.model.layers[int(s.split("_")[1])] for s in SITES
        }
        write_json(out / "model_manifest.json", observer.manifest())

        def budget(n=1):
            """Bound complete case blocks and actual forwards."""
            if (
                observer.calls + n > 1800
                or time.monotonic() - start > 900
                or torch.mps.current_allocated_memory() > 9 * 1024**3
            ):
                raise BudgetExceeded("P1C budget")

        tokens, spans, groups, states = {}, {}, {}, {}
        lookup = {(r["group_id"], r["variant"], r["level"]): r["case_id"] for r in core}
        for r in core + opportunity:
            case = r["case_id"]
            tokens[case] = observer.prompt_ids(SYSTEM, r["user"])
            spans[case] = observer.content_indices(SYSTEM, r["user"], r["evidence"])
            groups[case] = partitions(len(tokens[case]), spans[case])
            if any(
                len(observer.candidate_ids(tokens[case], r[k])) != 1
                for k in ("high_answer", "low_answer")
            ):
                raise ValueError("Frozen P1C candidates must be single-token")
        for r in core:
            case = r["case_id"]
            donor = lookup[r["group_id"], r["variant"], -r["level"]]
            check_donor_layout(tokens[case], spans[case], tokens[donor], spans[donor])
        write_json(
            out / "token_layouts.json",
            {
                k: {"ids": v, "span": spans[k], "partitions": groups[k]}
                for k, v in tokens.items()
            },
        )

        def record(row, logits):
            """Save raw candidate logits with their semantic orientation."""
            ids = tokens[row["case_id"]]
            high = observer.candidate_ids(ids, row["high_answer"])[0]
            low = observer.candidate_ids(ids, row["low_answer"])[0]
            return {
                k: row[k]
                for k in ("case_id", "group_id", "variant", "level", "condition")
            } | {
                "margin": float(logits[high]) - float(logits[low]),
                "high_logit": float(logits[high]),
                "low_logit": float(logits[low]),
            }

        for i, r in enumerate(core):
            budget()
            case = r["case_id"]
            captured, logits = observer.capture(
                tokens[case], positions=list(range(len(tokens[case])))
            )
            states[case] = captured
            np.savez_compressed(cache / (case + ".npz"), **captured)
            base.append(record(r, logits))
            if i == 0:
                for site in SITES:
                    budget(2)
                    _, same = observer.capture(
                        tokens[case],
                        patch=Patch(site, len(tokens[case]) - 1, captured[site][-1]),
                    )
                    _, restored = observer.capture(tokens[case])
                    error = max(
                        float(np.max(abs(same - logits))),
                        float(np.max(abs(restored - logits))),
                    )
                    if error > 1e-5:
                        raise ValueError("Instrument restore failed")
                    checks.append({"site": site, "self_restore": error})
            if (i + 1) % 12 == 0:
                flush("NATIVE_RUNNING")
                print(f"NATIVE {i + 1}/60 calls={observer.calls}", flush=True)
        for i, r in enumerate(core):
            budget(21)
            case = r["case_id"]
            donor = lookup[r["group_id"], r["variant"], -r["level"]]
            for site, mask in product(SITES, MASKS[1:]):
                indices = sorted(
                    i
                    for name, bit in zip("EQD", mask)
                    if bit
                    for i in groups[case][name]
                )
                budget()
                _, logits = observer.capture(
                    tokens[case],
                    patch=Patch(site, indices, states[donor][site][indices]),
                )
                effects.append(
                    record(r, logits)
                    | {
                        "site": site,
                        "mask": "".join(map(str, mask)),
                        "donor_case": donor,
                    }
                )
            if (i + 1) % 12 == 0:
                flush("INTERVENTIONS_RUNNING")
                print(f"INTERVENTIONS {i + 1}/60 calls={observer.calls}", flush=True)
        for r in opportunity:
            budget()
            _, logits = observer.capture(tokens[r["case_id"]])
            native.append(record(r, logits))
        tables = analyze(
            pd.DataFrame(base),
            pd.DataFrame(effects),
            pd.DataFrame(native),
            pd.read_csv(args.previous_dir / "baselines.csv"),
        )
        full = (
            tables["transport_rows"].query("mask == '111'").residual_error.abs().max()
        )
        if full > 1e-5:
            raise ValueError(f"Full donor restoration mismatch: {full}")
        for name, frame in tables.items():
            frame.to_csv(out / (name + ".csv"), index=False)
        result = flush(
            "COMPLETE",
            full_donor_restore_max=float(full),
            previous_identity=receipt["parent_identity"],
        )
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
    """Collect locally without changing or restarting any predecessor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-dir", type=Path, required=True)
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
