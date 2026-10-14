#!/usr/bin/env python
"""Structural, text-free characterization of conflicting duplicate-key rows (c2_dedup_v1 Stage I checkpoint).

Only aggregate relations are reported: prefix/substring containment, normalized equality, length ratio, token
Jaccard, file adjacency and venue class. No raw text is written.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("c2", REPO / "scripts/run_grain_c2_mcd_venue_dependence.py")
c2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c2)
OUT = REPO / "results/grain_gptpro_foundation_v3/P05/c2_dedup_sensitivity_r1"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def main() -> int:
    raw = pd.read_csv(c2.COMMENTS)
    raw["user_id"] = raw["user_id"].astype(str)
    raw["row"] = np.arange(len(raw))
    raw["body_raw"] = raw["body_raw"].fillna("").astype(str)
    g = raw[raw.duplicated(["user_id", "comment_id"], keep=False)].groupby(["user_id", "comment_id"], sort=False)
    rel = {"body_equal": 0, "body_equal_after_ws_case_norm": 0, "later_is_prefix_of_first": 0, "first_is_prefix_of_later": 0,
           "later_substring_of_first": 0, "first_substring_of_later": 0, "neither_contains": 0}
    ratios, jac, adjacent, trait_conf, trait_exact, conf_groups, conf_rows_by_copies = [], [], 0, 0, 0, set(), {}
    pairs = 0
    for key, grp in g:
        first = grp.iloc[0]
        for _, row in grp.iloc[1:].iterrows():
            a, b = first["body_raw"], row["body_raw"]
            pairs += 1
            is_conf = a != b
            if not is_conf:
                rel["body_equal"] += 1
                trait_exact += int(c2.is_trait_named(row["subreddit"]))
                continue
            conf_groups.add(key)
            conf_rows_by_copies[len(grp)] = conf_rows_by_copies.get(len(grp), 0) + 1
            trait_conf += int(c2.is_trait_named(row["subreddit"]))
            if norm(a) == norm(b):
                rel["body_equal_after_ws_case_norm"] += 1
            elif a.startswith(b):
                rel["later_is_prefix_of_first"] += 1
            elif b.startswith(a):
                rel["first_is_prefix_of_later"] += 1
            elif b in a:
                rel["later_substring_of_first"] += 1
            elif a in b:
                rel["first_substring_of_later"] += 1
            else:
                rel["neither_contains"] += 1
            ratios.append(len(b) / max(len(a), 1))
            ta, tb = set(norm(a).split()), set(norm(b).split())
            jac.append(len(ta & tb) / max(len(ta | tb), 1))
            adjacent += int(abs(int(row["row"]) - int(first["row"])) == 1)
    body_conf = sum(v for k, v in rel.items() if k != "body_equal")
    out = {
        "extra_row_pairs_checked": pairs,
        "body_relation_counts": rel,
        "body_conflicting_pairs": body_conf,
        "conflicting_keys": len(conf_groups),
        "conflicting_pairs_by_group_size": {str(k): v for k, v in sorted(conf_rows_by_copies.items())},
        "length_ratio_later_over_first": {"median": float(np.median(ratios)) if ratios else None,
                                           "p10": float(np.percentile(ratios, 10)) if ratios else None,
                                           "p90": float(np.percentile(ratios, 90)) if ratios else None},
        "token_jaccard": {"median": float(np.median(jac)) if jac else None, "p10": float(np.percentile(jac, 10)) if jac else None,
                          "share_below_.2": float(np.mean(np.array(jac) < 0.2)) if jac else None},
        "adjacent_in_file_share": adjacent / max(body_conf, 1),
        "trait_named_share_body_conflicting": trait_conf / max(body_conf, 1),
        "trait_named_share_body_equal": trait_exact / max(rel["body_equal"], 1),
        "note": "aggregate relations only; no raw text written",
    }
    (OUT / "inventory_conflict_structure.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
