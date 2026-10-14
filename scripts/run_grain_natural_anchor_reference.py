#!/usr/bin/env python3
"""Bounded local SIMPA anchor expansion on source-only, already encoded comments.

No questionnaire outcomes are read or used. A higher retrieval count is not a
precision or personality-validity gain. Private passages stay in ignored output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.evidence_contracts import verify_frozen_top1
from grain_research.natural_anchors import expand_bank, fingerprint, retrieve
from scripts.prepare_grain_predecessor_reference import DEFAULT, digest, write
from scripts.run_grain_paraphrase_reference import MODEL, REVISION, encode_batches


def run(args):
    """Freeze anchors first, compare retrieval and export distinct scope channels."""
    out = args.output
    if out.exists():
        raise ValueError("Use a fresh output")
    out.mkdir(parents=True)
    config = json.loads(args.anchors.read_text())
    anchors = expand_bank(config)
    write(
        out / "frozen_local_anchors.json",
        {"anchors": anchors, "sha256": fingerprint(anchors)},
    )
    write(
        out / "contract.json",
        {
            "anchors_sha256": fingerprint(anchors),
            "config_sha256": digest(args.anchors),
            "threshold": 0.6,
            "source_partition": "official fold != 0",
            "selection": "no personality outcomes, no threshold search",
            "scoring": "top1 per sentence per bank; anchor-scope channels separate, not verified source scope",
            "model": MODEL,
            "revision": REVISION,
            "human_expert_validation": False,
            "native_original_eTRS": False,
            "code_sha256": digest(Path(__file__)),
            "material_manifest_sha256": digest(
                args.material / "material_manifest.json"
            ),
            "max_review_pairs": 60,
        },
    )
    manifest = json.loads((args.material / "material_manifest.json").read_text())
    for name, sha in manifest["files"].items():
        if digest(args.material / name) != sha:
            raise ValueError("Material changed")
    data = json.loads((args.material / "private_material.json").read_text())
    with np.load(args.material / "private_arrays.npz", allow_pickle=False) as a:
        folds = a["folds"].copy()
    items = json.loads((args.material / "ipip_neo300.json").read_text())["items"]
    prior = pd.read_csv(args.producer / "private_sentence_evidence.csv")
    audit = verify_frozen_top1(args.material, args.producer, prior, data)
    write(out / "source_cache_verification.json", audit)
    blocks = []
    for path in sorted((args.producer / "paraphrase_cache").glob("batch_*.npz")):
        with np.load(path, allow_pickle=False) as z:
            blocks.append(z["vectors"])
    vectors = np.concatenate(blocks)
    q, h = vectors[: len(items)], vectors[len(items) :]
    indices = np.array(
        [i for i, s in enumerate(data["sentences"]) if folds[s["author_index"]] != 0]
    )
    sentences = [data["sentences"][i] for i in indices]
    h = h[indices]
    av, _ = encode_batches(
        [a["text"] for a in anchors],
        out / "anchor_cache",
        MODEL,
        REVISION,
        max_seconds=180,
    )
    existing = [
        {**i, "scope": "habitual", "provenance": "public_IPIP_item_not_natural_eTRS"}
        for i in items
    ]
    habitual = np.array([a["scope"] == "habitual" for a in anchors])
    episode = ~habitual
    banks = {
        "ipip300": (q, existing),
        "local_habitual40": (
            av[habitual],
            [a for a in anchors if a["scope"] == "habitual"],
        ),
        "local_episode20": (
            av[episode],
            [a for a in anchors if a["scope"] == "episode"],
        ),
        "ipip_plus_habitual340": (
            np.vstack([q, av[habitual]]),
            existing + [a for a in anchors if a["scope"] == "habitual"],
        ),
        "ipip_plus_local360": (np.vstack([q, av]), existing + anchors),
    }
    rows, summaries, accepted_by_bank = [], [], {}
    source_authors = set(np.flatnonzero(folds != 0).tolist())
    for name, (query, bank) in banks.items():
        ix, score, accepted = retrieve(h, query, bank)
        accepted_by_bank[name] = accepted
        matched_authors = set()
        for j in np.flatnonzero(accepted):
            s, a = sentences[j], bank[ix[j]]
            matched_authors.add(s["author_index"])
            rows.append(
                {
                    "bank": name,
                    "candidate_index": int(indices[j]),
                    **{
                        k: s[k]
                        for k in (
                            "author_index",
                            "comment_index",
                            "comment_id",
                            "start",
                            "end",
                        )
                    },
                    "anchor_id": a["id"],
                    "facet": a["facet"],
                    "trait": a["trait"],
                    "key": a["key"],
                    "anchor_scope": a["scope"],
                    "cosine": float(score[j]),
                    "anchor_text": a["text"],
                    "sentence": s["text"],
                    "provenance": a["provenance"],
                }
            )
        summaries.append(
            {
                "bank": name,
                "anchors": len(bank),
                "candidates": len(h),
                "matches": int(accepted.sum()),
                "matched_authors": len(matched_authors),
                "source_authors": len(source_authors),
                "author_coverage": len(matched_authors) / len(source_authors),
                "scope": "retrieval availability, not precision",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "private_matches.csv", index=False)
    frame.groupby(["bank", "trait", "anchor_scope", "key"]).agg(
        matches=("candidate_index", "size"), authors=("author_index", "nunique")
    ).to_csv(out / "coverage_by_scope.csv")
    # Address-based deduplication prevents treating repeated matches as new observations.
    frame.groupby(
        ["bank", "author_index", "trait", "anchor_scope", "key"]
    ).size().rename("count").to_csv(out / "private_author_evidence_channels.csv")
    pd.DataFrame(summaries).to_csv(out / "coverage.csv", index=False)
    old = accepted_by_bank["ipip300"]
    new = accepted_by_bank["ipip_plus_local360"]
    delta = {
        "retained_old_matches": int((old & new).sum()),
        "newly_detected": int((~old & new).sum()),
        "lost_detection": int((old & ~new).sum()),
        "larger_bank_multiple_comparison_effect": "not precision controlled",
    }
    old_rows = frame[frame.bank == "ipip300"].set_index("candidate_index")
    extended_rows = (
        frame[frame.bank == "ipip_plus_local360"]
        .set_index("candidate_index")
        .loc[old_rows.index]
    )
    delta["old_matches_reassigned_anchor"] = int(
        (old_rows.anchor_id != extended_rows.anchor_id).sum()
    )
    delta["old_matches_changed_trait_or_key"] = int(
        (
            (old_rows.trait != extended_rows.trait)
            | (old_rows.key != extended_rows.key)
        ).sum()
    )
    # Uniform stratified sample of new matches. Labels, cosine and outputs hidden.
    new_ids = set(indices[~old & new].tolist())
    pool = frame[
        (frame.bank == "ipip_plus_local360") & frame.candidate_index.isin(new_ids)
    ]
    packet, sampling = [], []
    for (trait, scope), group in pool.groupby(["trait", "anchor_scope"], sort=True):
        sample = group.sample(n=min(6, len(group)), random_state=731)
        sampling.append(
            {
                "trait": trait,
                "scope": scope,
                "pool_n": len(group),
                "sample_n": len(sample),
            }
        )
        for r in sample.itertuples():
            packet.append(
                {
                    "audit_id": f"KGA-{len(packet):03d}",
                    "sentence": r.sentence,
                    "hypothesis": r.anchor_text,
                    "expected_review_fields": [
                        "self_attribution",
                        "scope",
                        "relation",
                        "simpa_category",
                    ],
                }
            )
    write(
        out / "private_blind_review_packet.json",
        {
            "scope": "requires independent ratings; not automatically correct",
            "pairs": packet,
        },
    )
    write(out / "review_sampling.json", {"strata": sampling, "sampling_seed": 731})
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "coverage": summaries,
            "delta": delta,
            "local_anchors": len(anchors),
            "expert_anchors": 0,
            "feedback_promoted": 0,
            "review_pairs": len(packet),
            "personality_outcomes_used": False,
            "next": "independent natural-pair review before using expanded counts for target prediction",
        },
    )
    print(
        json.dumps(
            {"coverage": summaries, "delta": delta, "review_pairs": len(packet)},
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--material", type=Path, default=DEFAULT)
    p.add_argument("--producer", type=Path, default=DEFAULT.parent / "paraphrase_r1")
    p.add_argument(
        "--anchors", type=Path, default=ROOT / "configs/grain_natural_anchors_v1.json"
    )
    p.add_argument("--output", type=Path, required=True)
    with threadpool_limits(limits=2):
        run(p.parse_args())
