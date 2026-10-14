#!/usr/bin/env python3
"""Standard crossed effects for detector availability in actually sampled comments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.evidence_contracts import verify_frozen_top1
from scripts.prepare_grain_predecessor_reference import BASE, DEFAULT, digest, write


def run(args):
    """Fit only source-side observed detection, never human traits or unseen choices."""
    if args.output.exists():
        raise ValueError("Use a new output directory")
    args.output.mkdir(parents=True)
    manifest = json.loads((args.material / "material_manifest.json").read_text())
    if (
        digest(args.material / "private_material.json")
        != manifest["files"]["private_material.json"]
    ):
        raise ValueError("Changed source material")
    data = json.loads((args.material / "private_material.json").read_text())
    evidence = pd.read_csv(args.evidence)
    verified = verify_frozen_top1(args.material, args.evidence.parent, evidence, data)
    hits = set(
        zip(
            evidence.loc[evidence.accepted, "author_index"],
            evidence.loc[evidence.accepted, "comment_index"],
        )
    )
    rows = []
    for i, b in enumerate(data["bundles"]):
        if b["official_folds"]["big5"] == 0:
            continue
        if b["source_group"] != "big5_only":
            raise ValueError("Source-only material required")
        for j, c in enumerate([c for view in b["views"] for c in view]):
            if not c["subreddit"] or c["word_count"] <= 0:
                raise ValueError("Observed condition and exposure length required")
            rows.append(
                {
                    "author": str(i),
                    "condition": c["subreddit"],
                    "comment_index": j,
                    "words": c["word_count"],
                    "matched": int((i, j) in hits),
                }
            )
    frame = pd.DataFrame(rows)
    log = np.log(frame.words.to_numpy(float))
    frame["log_words_z"] = (log - log.mean()) / log.std()
    frame.to_csv(args.output / "private_observed_comments.csv", index=False)
    r_script = ROOT / "scripts/fit_grain_evidence_conditions.R"
    command = [
        "/usr/local/bin/Rscript",
        str(r_script),
        str(args.output / "private_observed_comments.csv"),
        str(args.output),
    ]
    result = subprocess.run(
        command, text=True, capture_output=True, timeout=120, check=False
    )
    (args.output / "r_output.txt").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("Availability mixed model failed; see r_output.txt")
    diagnostics = (
        pd.read_csv(args.output / "diagnostics.csv").fillna("").to_dict("records")[0]
    )
    summary = {
        "status": "COMPLETE",
        "evidence_contract_verification": verified,
        "observed_comments": len(frame),
        "authors": frame.author.nunique(),
        "conditions": frame.condition.nunique(),
        "matched_comments": int(frame.matched.sum()),
        "match_prevalence": float(frame.matched.mean()),
        "equation": "logit Pr(detector match in sampled comment)=beta0+betaL*z(log words)+a_author+b_selected_subreddit",
        "response": "at least one paraphrase top-1 cosine>=0.6 detection, not validated personality evidence",
        "fixed_effects": pd.read_csv(args.output / "fixed_effects.csv").to_dict(
            "records"
        ),
        "variance_components": pd.read_csv(args.output / "variance_components.csv")
        .fillna("")
        .to_dict("records"),
        "diagnostics": diagnostics,
        "observed_exposure": "sampled comments and words only",
        "unobserved_opportunity_set": "UNKNOWN; unvisited subreddits are not rejected choices",
        "personality_labels_loaded": False,
        "held_out_fold0_loaded_for_fit": False,
        "scope": "Descriptive standard GLMM on authorized source comments. Random author effects can contain style, conditions and detector error; they are not personality variance. AIC is not held-out performance.",
        "source_hashes": {
            "material": digest(args.material / "material_manifest.json"),
            "evidence": digest(args.evidence),
            "script": digest(Path(__file__)),
            "evidence_contract_code": digest(
                ROOT / "grain_research/evidence_contracts.py"
            ),
            "R_script": digest(r_script),
        },
    }
    write(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    """Use one fixed standard model, bounded to two minutes, without model search."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--material", type=Path, default=DEFAULT)
    p.add_argument(
        "--evidence",
        type=Path,
        default=BASE
        / "predecessor_integration_v1/paraphrase_r1/private_sentence_evidence.csv",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=BASE / "predecessor_integration_v1/observed_conditions_r1",
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
