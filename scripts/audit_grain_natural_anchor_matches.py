#!/usr/bin/env python3
"""Local NLI screening of new anchor matches; never substitutes for expert review."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_grain_predecessor_reference import digest, write
from scripts.run_grain_entailment_reference import MODEL, REVISION
from scripts.run_grain_evidence_calibration import infer_pairs


def run(args):
    """Preserve raw NLI decisions and unresolved long pairs without relabeling traits."""
    from transformers import AutoTokenizer

    if args.output.exists():
        raise ValueError("Use a fresh output directory")
    args.output.mkdir(parents=True)
    packet = json.loads(args.packet.read_text())
    frame = pd.DataFrame(packet["pairs"]).rename(columns={"hypothesis": "item_text"})
    if frame.audit_id.duplicated().any():
        raise ValueError("Duplicate packet identities")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True, trust_remote_code=False
    )
    lengths = tokenizer(
        frame.sentence.tolist(), frame.item_text.tolist(), truncation=False
    )["input_ids"]
    frame["pair_tokens"] = [len(t) for t in lengths]
    eligible = frame.pair_tokens <= 512
    result = infer_pairs(frame[eligible])
    frame.merge(result, on="audit_id", how="left", validate="one_to_one").to_csv(
        args.output / "private_local_nli_screen.csv", index=False
    )
    write(
        args.output / "summary.json",
        {
            "status": "COMPLETE",
            "packet_sha256": digest(args.packet),
            "model": MODEL,
            "revision": REVISION,
            "total_pairs": len(frame),
            "evaluated_pairs": int(eligible.sum()),
            "too_long": int((~eligible).sum()),
            "nli_counts": result.nli.value_counts().to_dict(),
            "precision": None,
            "expert_reviewed": False,
            "feedback_promoted": 0,
            "scope": "independent fixed NLI model screening, no semantic gold or personality validity",
        },
    )
    print((args.output / "summary.json").read_text())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
