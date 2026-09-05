#!/usr/bin/env python3
"""Count VM05 label-source coverage without loading any psychological values."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.cohort import ID_SEED
from grain_research.vector_mechanisms.data_views import stable_key
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from scripts.run_grain_vector_mechanism_controls import write_json


def main():
    """Separate official membership from observed outcomes and model selection."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--cohort-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a new inventory directory")
    archive_source(ROOT, out / "source_snapshot.zip")
    paths = {
        "big5": a.data_root / "pandora_official_big5_prepared.csv",
        "bridge": a.data_root / "bridge/pandora_official_bridge_strict377.csv",
    }
    paths.update(
        {
            axis: a.data_root / "mbti_axes" / f"pandora_official_{axis}_prepared.csv"
            for axis in ("EI_cont", "SN_cont", "TF_cont", "JP_cont")
        }
    )
    sets, metadata = {}, {}
    for name, path in paths.items():
        frame = pd.read_csv(path, usecols=["user_id"], dtype={"user_id": str})
        if frame.user_id.isna().any() or frame.user_id.duplicated().any():
            raise ValueError("Missing or repeated official author IDs")
        sets[name] = {stable_key(u, ID_SEED)[:24] for u in frame.user_id}
        metadata[name] = {
            "relative_path": str(path.relative_to(a.data_root)),
            "authors": len(frame),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "columns_loaded": ["user_id"],
        }
    mbti_union = set().union(*(sets[k] for k in sets if k.endswith("_cont")))
    sources = {
        "big5_all": sets["big5"],
        "big5_only": sets["big5"] - mbti_union,
        "bridge_strict_report": sets["bridge"],
        "any_both_membership": sets["big5"] & mbti_union,
    }
    sources.update(
        {k + "_only": v - sets["big5"] for k, v in sets.items() if k.endswith("_cont")}
    )
    rows = []
    cohort_ids = set()
    for split in ("discovery", "calibration", "mechanism_holdout"):
        views = json.loads(
            (a.cohort_dir / f"{split}_views.json").read_text(encoding="utf-8")
        )
        ids = {v["author_id"] for v in views}
        if cohort_ids & ids or len(ids) != len(views):
            raise ValueError("Cohort author overlap")
        cohort_ids |= ids
        for name, members in sources.items():
            rows.append(
                {
                    "split": split,
                    "source": name,
                    "authors": len(ids & members),
                    "cohort_authors": len(ids),
                }
            )
    for name, members in sources.items():
        rows.append(
            {
                "split": "all",
                "source": name,
                "authors": len(cohort_ids & members),
                "cohort_authors": len(cohort_ids),
            }
        )
    pd.DataFrame(rows).to_csv(out / "coverage.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "files": metadata,
        "source_counts": {k: len(v) for k, v in sources.items()},
        "cohort_authors": len(cohort_ids),
        "psychological_values_loaded": False,
        "membership_not_trait_completeness": True,
        "code_sha256": code_fingerprint(ROOT),
        "warning": "Only membership coverage; no class prevalence, label quality, effect, or power calculation",
    }
    write_json(out / "summary.json", summary)
    print(pd.DataFrame(rows).query("split == 'all'").to_string(index=False))


if __name__ == "__main__":
    main()
