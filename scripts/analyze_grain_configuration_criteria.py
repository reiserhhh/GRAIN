#!/usr/bin/env python3
"""Join frozen configuration observations to permitted source author criteria."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.criterion_accounting import criterion_account
from grain_research.vector_mechanisms.direct_traits import AXES, BIG5
from grain_research.vector_mechanisms.factor_catalog import association, bh_adjust
from grain_research.vector_mechanisms.natural_transport import paired_summary
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write

PLAN = ROOT / "docs/GRAIN_LOGIC_CRITERION_CONNECTION_PLAN_V1.md"
SOURCE = BASE / "logic_configuration_extraction_v1"
REL = BASE / "logic_configuration_relations_v1/operator_aligned_r3"
CRITERIA = BASE / "all_factor_source_m29_v1/private_external_links.csv"


def checked_join(values, criteria):
    """Require one exact criterion per author/query; never silently lose rows."""
    keys = ["author_id", "query"]
    if values.duplicated(keys).any() or criteria.duplicated(keys).any():
        raise ValueError("Duplicate author-query observation")
    joined = values.merge(
        criteria[keys + ["y_true"]], on=keys, how="left", validate="one_to_one"
    )
    if len(joined) != len(values) or not np.isfinite(joined.y_true).all():
        raise ValueError("Missing criterion join")
    return joined.sort_values(keys)


def link(query, x, y, bootstrap=False):
    """Keep unflipped binary scores and fixed-output author uncertainty."""
    if not np.isfinite([x, y]).all():
        raise ValueError("Finite observations required")
    stat = association(x, y, binary=query in AXES)
    if query in AXES:
        stat["minority_count"] = int(
            min(np.sum(np.asarray(y) == 0), np.sum(np.asarray(y) == 1))
        )
    if bootstrap:
        boot = paired_summary(y, x, calibrated=False)
        stat.update(
            {
                "bootstrap_" + k: v
                for k, v in boot.items()
                if k in ("r_lower", "r_upper", "valid_bootstraps")
            }
        )
    return stat


def run(out):
    """Compute association allocation only; no criteria are used to fit a model."""
    required = [
        PLAN,
        Path(__file__),
        CRITERIA,
        SOURCE / "private_source_oof.csv",
        REL / "private_query_configuration_views.csv",
    ]
    manifest = {
        "hashes": {str(p.relative_to(ROOT)): sha(p) for p in required},
        "new_model_fits": 0,
        "new_neural_calls": 0,
    }
    if (
        sha(SOURCE / "private_source_oof.csv")
        != read(SOURCE / "prediction_freeze.json")["source_sha256"]
    ):
        raise ValueError("Frozen source predictions changed")
    criteria = pd.read_csv(CRITERIA, dtype={"author_id": str})
    if (
        set(criteria["query"]) != {*BIG5, *AXES}
        or not criteria.y_true.between(0, 1).all()
    ):
        raise ValueError("Unexpected criterion coding")
    observations = pd.read_csv(REL / "private_query_configuration_views.csv")
    means = (
        observations.groupby(["author_id", "query"])[
            ["configuration", "grain", "residual"]
        ]
        .mean()
        .reset_index()
    )
    if (
        len(means) != 1317
        or observations.groupby(["author_id", "query"]).size().ne(3).any()
    ):
        raise ValueError("Exact276-author/three-view panel required")
    data = checked_join(means, criteria)
    data.to_csv(out / "private_teacher_criterion_links.csv", index=False)
    links, accounting = [], []
    max_error = 0.0
    for q, part in data.groupby("query"):
        y = part.y_true.to_numpy()
        prior = (
            criteria[criteria["query"] == q].set_index("author_id").loc[part.author_id]
        )
        if not np.allclose(prior.configuration, part.configuration, atol=1e-12, rtol=0):
            raise ValueError("Criterion-source and current configuration disagree")
        for channel in ("score", "prediction"):
            links.append(
                {
                    "scope": "source_whole_context",
                    "query": q,
                    "arm": "frozen_reader",
                    "channel": channel,
                    **link(q, prior[channel], y),
                }
            )
        parts = prior[["mean", "configuration", "final"]].to_numpy()
        parts = np.column_stack([parts, prior.score.to_numpy() - parts.sum(1)])
        terms = criterion_account(prior.score, parts, y)
        max_error = max(max_error, terms["closure_error"])
        for channel, value in zip(
            ("mean", "configuration", "final", "anchor"),
            terms["contributions"],
            strict=True,
        ):
            accounting.append(
                {
                    "scope": "source_whole_function",
                    "query": q,
                    "arm": "frozen_reader",
                    "target": "score",
                    "channel": channel,
                    "n": len(part),
                    "total_r": terms["total_r"],
                    "r_contribution": value,
                }
            )
        for channel in ("configuration", "grain", "residual"):
            links.append(
                {
                    "scope": "source_teacher",
                    "query": q,
                    "arm": "full_source_reference",
                    "channel": channel,
                    **link(q, part[channel], y, channel != "grain"),
                }
            )
        terms = criterion_account(part.configuration, part[["grain", "residual"]], y)
        max_error = max(max_error, terms["closure_error"])
        for channel, value in zip(
            ("grain", "residual"), terms["contributions"], strict=True
        ):
            accounting.append(
                {
                    "scope": "source_teacher",
                    "query": q,
                    "arm": "full_source_reference",
                    "target": "configuration",
                    "channel": channel,
                    "n": len(part),
                    "total_r": terms["total_r"],
                    "r_contribution": value,
                }
            )
    oof = pd.read_csv(SOURCE / "private_source_oof.csv")
    private = []
    for (q, arm), group in oof.groupby(["query", "arm"]):
        if (
            group.groupby("author_id").fold.nunique().ne(1).any()
            or group.groupby("author_id").size().ne(3).any()
        ):
            raise ValueError("Author-fold grouping failed")
        p = (
            group.groupby(["author_id", "query"])[
                [
                    "actual_c",
                    "actual_r",
                    "predicted_c",
                    "predicted_r",
                    "derived_r",
                    "grain_part",
                ]
            ]
            .mean()
            .reset_index()
        )
        p = checked_join(p, criteria)
        original = data[data["query"] == q].set_index("author_id").loc[p.author_id]
        if not np.allclose(original.configuration, p.actual_c, atol=1e-12, rtol=0):
            raise ValueError("OOF and source configuration address differ")
        p["grain_proxy"] = p.predicted_c - p.predicted_r
        p["grain_transport_gap"] = p.grain_part - p.grain_proxy
        p["arm"] = arm
        private.append(p)
        for channel in (
            "actual_c",
            "actual_r",
            "predicted_c",
            "predicted_r",
            "derived_r",
            "grain_part",
            "grain_proxy",
            "grain_transport_gap",
        ):
            links.append(
                {
                    "scope": "source_oof",
                    "query": q,
                    "arm": arm,
                    "channel": channel,
                    **link(
                        q,
                        p[channel],
                        p.y_true,
                        arm == "e5_mean"
                        and channel
                        in ("actual_c", "actual_r", "predicted_c", "predicted_r"),
                    ),
                }
            )
        for target, prediction in (
            ("actual_c", "predicted_c"),
            ("actual_r", "predicted_r"),
            ("actual_r", "derived_r"),
        ):
            terms = criterion_account(
                p[target],
                np.column_stack([p[prediction], p[target] - p[prediction]]),
                p.y_true,
            )
            max_error = max(max_error, terms["closure_error"])
            for channel, value in zip(
                ("retained", "readout_error"), terms["contributions"], strict=True
            ):
                accounting.append(
                    {
                        "scope": "source_oof",
                        "query": q,
                        "arm": arm,
                        "target": target,
                        "estimate": prediction,
                        "channel": channel,
                        "n": len(p),
                        "total_r": terms["total_r"],
                        "r_contribution": value,
                    }
                )
    pd.concat(private, ignore_index=True).to_csv(
        out / "private_oof_criterion_links.csv", index=False
    )
    frame = pd.DataFrame(links)
    frame["hypothesis_id"] = [
        f"{r.scope}::{r.query}::{r.channel}::{('reference' if r.scope == 'source_oof' and r.channel in ('actual_c', 'actual_r', 'grain_part') else r.arm)}"
        for r in frame.itertuples()
    ]
    unique = frame.drop_duplicates("hypothesis_id").copy()
    unique["q_bh"] = unique.groupby("scope").p.transform(bh_adjust)
    frame["q_bh"] = frame.hypothesis_id.map(unique.set_index("hypothesis_id").q_bh)
    frame.to_csv(out / "criterion_links.csv", index=False)
    pd.DataFrame(accounting).to_csv(
        out / "criterion_covariance_accounting.csv", index=False
    )
    if max_error > 1e-10:
        raise ValueError("Criterion covariance closure failed")
    manifest.update(
        association_rows=len(frame),
        covariance_rows=len(accounting),
        max_covariance_error=max_error,
        author_counts={"big5_only": 213, "mbti_only": 63},
        private_labels_used_for_fitting=False,
        distinct_tests_by_scope=unique.groupby("scope").size().to_dict(),
    )
    write(out / "manifest.json", manifest)
    print(
        frame.query(
            "scope=='source_teacher' or (scope=='source_oof' and arm=='e5_mean')"
        )[
            [
                "scope",
                "query",
                "channel",
                "n",
                "pearson_r",
                "spearman_rho",
                "auc",
                "q_bh",
            ]
        ].to_string(index=False),
        flush=True,
    )


def main():
    """Preserve every exploratory packet in a new private results directory."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir", type=Path, default=BASE / "logic_configuration_criterion_v1"
    )
    args = p.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE):
        raise ValueError("Private results directory required")
    out.mkdir(parents=True, exist_ok=False)
    write(out / "status.json", {"status": "RUNNING"})
    try:
        run(out)
        write(out / "status.json", {"status": "COMPLETE"})
    except Exception as exc:
        write(
            out / "status.json", {"status": "FAILED", "error_type": type(exc).__name__}
        )
        raise


if __name__ == "__main__":
    main()
