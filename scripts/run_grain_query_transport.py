#!/usr/bin/env python3
"""Localize Q transport into four vector-position bins and compare compact operators."""

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
from grain_research.vector_mechanisms.trace import DecoderObserver, Patch
from scripts.audit_grain_evidence_conduit import audit
from scripts.run_grain_persona_reciprocity import (
    EXPECTED_METADATA,
    BudgetExceeded,
    file_hash,
)
from scripts.run_grain_vector_mechanism_controls import write_json

MASKS = list(product((0, 1), repeat=4))


def bin_indices(indices):
    """Split fixed ordered query positions without reading their linguistic content."""
    if (
        len(indices) < 4
        or len(set(indices)) != len(indices)
        or sorted(indices) != indices
    ):
        raise ValueError("Four nonempty ordered bins required")
    return [a.tolist() for a in np.array_split(indices, 4)]


def design(masks, degree):
    """Binary main/pair/all interactions without an intercept: no patch has zero effect."""
    x = np.asarray(masks, dtype=float)
    cols = [tuple(c) for n in range(1, degree + 1) for c in combinations(range(4), n)]
    return np.column_stack([np.prod(x[:, c], axis=1) for c in cols]), cols


def analyze(rows):
    """Fit on3 settings and report the other3, with the donor-gap oracle made explicit."""
    df = pd.DataFrame(rows)
    fit_groups = sorted(df.group_id.unique())[:3]
    results, predictions, terms = [], [], []
    for case, sub in df.groupby("case_id"):
        sub = (
            sub.set_index("mask")
            .loc[["".join(map(str, m)) for m in MASKS]]
            .reset_index()
        )
        x, cols = design(MASKS, 4)
        beta = np.linalg.lstsq(x, sub["shift"].to_numpy(), rcond=None)[0]
        for term, b in zip(cols, beta):
            terms.append(
                {
                    "case_id": case,
                    "group_id": sub.group_id.iloc[0],
                    "variant": sub.variant.iloc[0],
                    "term": "".join(str(i + 1) for i in term),
                    "coefficient": float(b),
                }
            )
    for variant in ("event_target", "trait_code1"):
        sub = df[df.variant == variant].copy()
        if (sub.donor_gap.abs() < 1e-8).any():
            raise ValueError("Unstable oracle normalization; no case dropping")
        train = sub.group_id.isin(fit_groups).to_numpy()
        y = sub["shift"].to_numpy() / sub.donor_gap.to_numpy()
        masks = [list(map(int, m)) for m in sub["mask"]]
        for degree in (1, 2):
            x, cols = design(masks, degree)
            beta = np.linalg.lstsq(x[train], y[train], rcond=None)[0]
            predicted = (x @ beta) * sub.donor_gap.to_numpy()
            for split, sel in (("fit", train), ("heldout", ~train)):
                error = predicted[sel] - sub["shift"].to_numpy()[sel]
                nochange = sub["shift"].to_numpy()[sel]
                results.append(
                    {
                        "variant": variant,
                        "degree": degree,
                        "split": split,
                        "cases": int(sel.sum()),
                        "scenario_count": int(sub.loc[sel, "group_id"].nunique()),
                        "rmse": float(np.sqrt(np.mean(error**2))),
                        "nochange_rmse": float(np.sqrt(np.mean(nochange**2))),
                        "relative_error_energy": float(
                            np.sum(error**2) / np.sum(nochange**2)
                        ),
                        "oracle_donor_gap_used": True,
                        "coefficients": json.dumps(beta.tolist()),
                    }
                )
            for i, r in enumerate(sub.to_dict("records")):
                predictions.append(
                    {
                        **r,
                        "degree": degree,
                        "split": "fit" if train[i] else "heldout",
                        "predicted_shift": float(predicted[i]),
                    }
                )
    return {
        "operator_scores": pd.DataFrame(results),
        "operator_predictions": pd.DataFrame(predictions),
        "interaction_terms": pd.DataFrame(terms),
    }


def run(args):
    """Apply only stored counterfactual vectors from the verified parent packet."""
    receipt, _ = audit(args.parent_dir, args.p1b_dir)
    parent_manifest = json.loads((args.parent_dir / "manifest.json").read_text())
    material = json.loads((args.parent_dir / "materials.json").read_text())["core"]
    selected = [
        r for r in material if r["variant"] in ("event_target", "trait_code1", "venue")
    ]
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__),
        ROOT / "scripts/audit_grain_evidence_conduit.py",
        ROOT / "grain_research/vector_mechanisms/trace.py",
        ROOT / "scripts/run_grain_persona_reciprocity.py",
        ROOT / "grain_research/vector_mechanisms/controls.py",
        ROOT / "scripts/run_grain_vector_mechanism_controls.py",
        ROOT / "docs/GRAIN_QUERY_TRANSPORT_P1D_PROTOCOL.md",
        ROOT / "tests/test_grain_query_transport.py",
    ]
    manifest = {
        "parent_identity": receipt["parent_identity"],
        "source_material_hash": parent_manifest["material_hash"],
        "code": {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
        "max_calls": 800,
        "max_seconds": 600,
        "private_data_read": False,
        "site": "block_16",
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    with zipfile.ZipFile(out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for path in sources:
            z.write(path, path.relative_to(ROOT))
    base = pd.read_csv(args.parent_dir / "baselines.csv").set_index("case_id")
    prior = pd.read_csv(args.parent_dir / "interventions.csv", dtype={"mask": str})
    prior = prior[(prior["mask"] == "010") & (prior.site == "block_16")].set_index(
        "case_id"
    )
    layouts = json.loads((args.parent_dir / "token_layouts.json").read_text())
    bins = {
        r["case_id"]: bin_indices(layouts[r["case_id"]]["partitions"]["Q"])
        for r in selected
    }
    write_json(out / "bins.json", bins)
    lookup = {(r["group_id"], r["variant"], r["level"]): r["case_id"] for r in selected}
    states = {
        r["case_id"]: np.load(
            args.parent_dir / "activations" / (r["case_id"] + ".npz"),
            allow_pickle=False,
        )["block_16"]
        for r in selected
    }
    rows, checks = [], []
    observer, start = None, time.monotonic()
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )

    def flush(status, **extra):
        """Keep all36x16 rows including zero and failed-run status."""
        pd.DataFrame(rows).to_csv(out / "scores.csv", index=False)
        write_json(out / "checks.json", checks)
        value = {
            "status": status,
            "identity": manifest["identity"],
            "calls": observer.calls if observer else 0,
            "seconds": time.monotonic() - start,
            "rows": len(rows),
            **extra,
        }
        write_json(out / "run_status.json", value)
        return value

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
        observer.sites = {"block_16": observer.sites["block_16"]}
        write_json(out / "model_manifest.json", observer.manifest())

        def budget(n=1):
            """Bound paired complete cases and actual calls."""
            if (
                observer.calls + n > 800
                or time.monotonic() - start > 600
                or torch.mps.current_allocated_memory() > 9 * 1024**3
            ):
                raise BudgetExceeded("P1D budget")

        def score(r, patch=None):
            """Score the unchanged semantic candidates, preserving full native logits."""
            budget()
            ids = layouts[r["case_id"]]["ids"]
            hi = observer.candidate_ids(ids, r["high_answer"])
            lo = observer.candidate_ids(ids, r["low_answer"])
            if len(hi) != 1 or len(lo) != 1:
                raise ValueError("Single tokens required")
            _, logits = observer.capture(ids, patch=patch)
            return float(logits[hi[0]]) - float(logits[lo[0]])

        for i, r in enumerate(selected):
            budget(17)
            case = r["case_id"]
            donor = lookup[r["group_id"], r["variant"], -r["level"]]
            if i < 3:
                native = score(r)
                indices = sorted(j for block in bins[case] for j in block)
                copied = score(r, Patch("block_16", indices, states[case][indices]))
                err = max(abs(native - base.loc[case, "margin"]), abs(copied - native))
                checks.append({"case_id": case, "restore_error": err})
                if err > 1e-5:
                    raise ValueError("Native/self-copy mismatch")
            for mask in MASKS:
                indices = sorted(
                    j for block, bit in zip(bins[case], mask) if bit for j in block
                )
                margin = (
                    score(r, Patch("block_16", indices, states[donor][indices]))
                    if indices
                    else float(base.loc[case, "margin"])
                )
                if mask == (1, 1, 1, 1):
                    err = abs(margin - prior.loc[case, "margin"])
                    checks.append({"case_id": case, "parent_q_patch_error": err})
                    if err > 1e-5:
                        raise ValueError("Parent Q patch not reproduced")
                rows.append(
                    {
                        "case_id": case,
                        "group_id": r["group_id"],
                        "variant": r["variant"],
                        "level": r["level"],
                        "mask": "".join(map(str, mask)),
                        "margin": margin,
                        "shift": margin - base.loc[case, "margin"],
                        "donor_gap": base.loc[donor, "margin"]
                        - base.loc[case, "margin"],
                        "donor_case": donor,
                    }
                )
            if (i + 1) % 6 == 0:
                flush("RUNNING", completed_cases=i + 1)
                print(f"QUERY_BINS {i + 1}/36 calls={observer.calls}", flush=True)
        for name, table in analyze(rows).items():
            table.to_csv(out / (name + ".csv"), index=False)
        result = flush(
            "COMPLETE", new_personality_prediction=False, oracle_donor_gap_used=True
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
    except Exception as e:
        flush(
            "PARTIAL_BUDGET" if isinstance(e, BudgetExceeded) else "FAILED",
            error=str(e),
        )
        (out / "error_log.txt").write_text(traceback.format_exc())
        raise


def main():
    """Run only the frozen follow-up on local public checkpoint and synthetic caches."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-dir", type=Path, required=True)
    p.add_argument("--p1b-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT
        / "results/grain_vector_mechanisms/local_observers/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
