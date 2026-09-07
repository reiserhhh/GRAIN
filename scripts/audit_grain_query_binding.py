#!/usr/bin/env python3
"""Independent complete-factorial and operand-offset audit for P1I."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    """Compute an immutable local artifact identity."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(result, parent, model_dir):
    """Reconstruct offset coverage, controls and scalar contrasts without neural inference."""
    manifest = json.loads((result / "manifest.json").read_text())
    status = json.loads((result / "run_status.json").read_text())
    if status["status"] != "COMPLETE" or status["calls"] != 388:
        raise ValueError("Incomplete P1I")
    for directory in (result, parent):
        for name, expected in json.loads(
            (directory / "output_hashes.json").read_text()
        ).items():
            if sha(directory / name) != expected:
                raise ValueError("Artifact changed")
    with zipfile.ZipFile(result / "source_snapshot.zip") as z:
        for name, expected in manifest["code"].items():
            if hashlib.sha256(z.read(name)).hexdigest() != expected:
                raise ValueError("Frozen source changed")
    if (
        manifest["parent_identity"]
        != json.loads((parent / "manifest.json").read_text())["identity"]
    ):
        raise ValueError("Different parent")
    scores = pd.read_csv(result / "scores.csv", dtype={"mask": str})
    terms = pd.read_csv(result / "interaction_terms.csv")
    summary = pd.read_csv(result / "transport_summary.csv", dtype={"mask": str})
    native = pd.read_csv(parent / "baselines.csv").set_index("case_id").margin
    parent_q = (
        pd.read_csv(parent / "effects.csv")
        .query("patch == 'Q'")
        .set_index("case_id")
        .margin
    )
    if len(scores) != 480 or scores.duplicated(["case_id", "mask"]).any():
        raise ValueError("Wrong grid")
    masks = list(product((0, 1), repeat=3))
    names = ["".join(map(str, m)) for m in masks]
    cols = [c for n in (1, 2, 3) for c in combinations(range(3), n)]
    design = np.array([[np.prod([m[j] for j in c]) for c in cols] for m in masks])
    coefficient_error, endpoint_error = 0.0, 0.0
    for case, sub in scores.groupby("case_id"):
        if set(sub["mask"]) != set(names + ["random_N", "random_K"]):
            raise ValueError("Missing factorial/control")
        sub = sub.set_index("mask")
        endpoint_error = max(
            endpoint_error,
            abs(sub.loc["000", "margin"] - native[case]),
            abs(sub.loc["111", "margin"] - parent_q[case]),
        )
        if (sub.loc[["000", "111"], "source"] != "parent").any() or (
            sub.drop(["000", "111"]).source != "new"
        ).any():
            raise ValueError("Endpoint provenance differs")
        coefficients = np.linalg.lstsq(
            design, sub.loc[names, "margin"].to_numpy() - native[case], rcond=None
        )[0]
        saved = terms[terms.case_id == case].set_index("term").value
        for c, value in zip(cols, coefficients):
            coefficient_error = max(
                coefficient_error, abs(value - saved["".join("NKR"[j] for j in c)])
            )
    data = json.loads((parent / "materials.json").read_text())
    layouts = json.loads((parent / "token_layouts.json").read_text())
    regions = json.loads((result / "regions.json").read_text())
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False
    )
    for row in data["rows"]:
        case = row["case_id"]
        item, layout = regions[case], layouts[case]
        text = tok.apply_chat_template(
            [
                {"role": "system", "content": data["system"]},
                {"role": "user", "content": row["user"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoding = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        start = text.index(row["user"])
        if list(encoding["input_ids"]) != layout["ids"]:
            raise ValueError("Different tokenizer")
        for key, desc in item["intervals"].items():
            a, b = desc["span"]
            if row["user"][a:b].strip() != desc["text"]:
                raise ValueError("Named operand changed")
            expected = [
                j
                for j, (lo, hi) in enumerate(encoding["offset_mapping"])
                if start + a <= lo < hi <= start + b
            ]
            if expected != item["parts"][key]:
                raise ValueError("Operand token coverage")
            identity = (
                "__".join(str(row[k]) for k in ("group_id", "variant", "form")) + key
            )
            seed = int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)
            pool = sorted(set(layout["partitions"]["Q"]) - set(expected))
            control = sorted(
                np.random.default_rng(seed)
                .choice(pool, len(expected), replace=False)
                .tolist()
            )
            if control != item["controls"]["random_" + key]:
                raise ValueError("Count-matched random changed")
        if (
            sorted(j for part in item["parts"].values() for j in part)
            != layout["partitions"]["Q"]
        ):
            raise ValueError("Partition gap/overlap")
    arithmetic = float(abs(scores.margin - scores.high_logit + scores.low_logit).max())
    metric_error = 0.0
    for row in summary.itertuples():
        sub = scores[
            (scores.variant == row.variant)
            & (scores.form == row.form)
            & (scores["mask"] == row.mask)
        ]
        gap = sub.donor_case.map(native).to_numpy() - sub.case_id.map(native).to_numpy()
        error = sub.margin.to_numpy() - sub.donor_case.map(native).to_numpy()
        metric_error = max(
            metric_error,
            abs(np.dot(error, error) / np.dot(gap, gap) - row.relative_error_energy),
        )
    if max(arithmetic, coefficient_error, endpoint_error, metric_error) > 1e-8:
        raise ValueError("Scalar replay failed")
    return {
        "status": "VERIFIED",
        "parent_identity": manifest["identity"],
        "new_neural_calls": 0,
        "cases": scores.case_id.nunique(),
        "arithmetic_max": arithmetic,
        "coefficient_replay_max": coefficient_error,
        "reused_endpoint_max": endpoint_error,
        "metric_replay_max": metric_error,
        "interpretation": "complete neural intervention accounting, not unique psychological causation",
    }


def main():
    """Save one independent receipt with audit source, retaining original outputs."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-dir", type=Path, required=True)
    p.add_argument("--parent-dir", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.result_dir, args.parent_dir, args.model_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "audit_source.py").write_bytes(Path(__file__).read_bytes())
    result["source_sha256"] = sha(Path(__file__))
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
