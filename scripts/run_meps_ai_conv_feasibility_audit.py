#!/usr/bin/env python
"""Run a privacy-preserving SUICA feasibility audit on MEPS + AI conversations.

The script imports user text only in process memory, writes no raw text, and
does not produce participant-level mood records. It establishes whether the
protocol can support a *same-session, fixed-condition* SUICA pilot. It does not
estimate a stable personality trait or a V6 dynamic response operator.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from suica_core.meps_ai_conv import load_meps_ai_conv, public_metadata  # noqa: E402
from suica_core.meps_layout import reference_registry, resolve_meps_experiment_layout  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True,
                        help="Complete private bundle or direct decrypted_output root; never copied into this repo.")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/meps_ai_conv_feasibility_audit")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "reports/MEPS_AI_CONV_FEASIBILITY_AUDIT.md")
    parser.add_argument("--include-mood", action="store_true",
                        help="Explicit state-table opt-in; requires a separately authorized analysis.")
    return parser.parse_args()


def _view_summary(views: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (view_type, condition), group in views.groupby(["view_type", "condition"], dropna=False, sort=True):
        chars = group["user_chars"].to_numpy(float)
        rows.append({
            "view_type": view_type,
            "condition": condition,
            "n_views": int(len(group)),
            "n_participants": int(group["participant_code"].nunique()),
            "total_user_chars": int(chars.sum()),
            "median_user_chars": float(np.median(chars)),
            "minimum_user_chars": int(chars.min()),
            "maximum_user_chars": int(chars.max()),
            "median_user_turns": float(np.median(group["user_turns"])),
            "median_assistant_turns": float(np.median(group["assistant_turns"])),
        })
    return pd.DataFrame(rows)


def _mood_summary(mood: pd.DataFrame) -> pd.DataFrame:
    return (
        mood.groupby(["phase", "source_scale"], sort=True)
        .agg(
            n_participants=("participant_code", "nunique"),
            n_items=("n_items", "median"),
            mean_response=("mean_response", "mean"),
            sd_between_participants=("mean_response", "std"),
        )
        .reset_index()
    )


def _direct_answer_summary(qc: pd.DataFrame) -> pd.DataFrame:
    return (
        qc.groupby("question_index", sort=True)
        .agg(
            n_expected=("participant_code", "size"),
            n_recovered=("recovered", "sum"),
            n_three_slot_format=("recognized_label_count", lambda values: int((values == 3).sum())),
            median_reported_chars=("reported_answer_chars", "median"),
            median_recovered_chars=("recovered_answer_chars", "median"),
            median_reported_minus_recovered=("reported_minus_recovered", "median"),
        )
        .reset_index()
    )


def _readiness(views: pd.DataFrame, mood: pd.DataFrame, qc: pd.DataFrame, metadata: dict) -> dict[str, object]:
    direct = views.loc[views["view_type"].eq("meps_answer")]
    free = views.loc[views["view_type"].eq("free_ai_chat")]
    complete_mood = (
        mood.groupby("participant_code")["phase"].nunique().eq(4).sum()
        if not mood.empty else 0
    )
    all_direct = bool(len(direct) == 3 * metadata["n_loaded_participants"])
    structured = bool((qc["recognized_label_count"] == 3).all()) if not qc.empty else False
    return {
        "verdict": "PILOT_READY_WITH_BOUNDARIES" if all_direct and structured else "IMPORT_REPAIR_REQUIRED",
        "supports": {
            "same_session_fixed_prompt_style_comparison": all_direct,
            "fixed_vs_free_condition_contrast": bool(len(free) == metadata["n_loaded_participants"]),
            "exploratory_text_to_state_anchor_analysis": int(complete_mood) == metadata["n_loaded_participants"],
            "protocol_or_opportunity_covariates": True,
        },
        "not_supported_by_this_dataset": [
            "cross_day_or_cross_session_retest",
            "stable_personality_trait_claim",
            "V6_two_epoch_by_two_replica_dynamic_estimator",
            "clinical_individual_scoring",
        ],
        "reasons": [
            "All observed text comes from one experimental session per participant.",
            "The three MEPS prompts are shared but distinct conditions, not technical replicas.",
            "Assistant turns are preserved only as context/opportunity metadata and never scored as participant text.",
            "Japanese text needs a frozen multilingual or Japanese representation layer before a V6 embedding/path analysis.",
        ],
    }


def _report(
    view_summary: pd.DataFrame,
    mood_summary: pd.DataFrame,
    direct_summary: pd.DataFrame,
    readiness: dict[str, object],
    metadata: dict[str, object],
) -> str:
    verdict = readiness["verdict"]
    return f"""# MEPS + AI Conversation: SUICA Feasibility Audit

## Decision

**{verdict}**. This private 46-participant corpus is suitable for a bounded
SUICA protocol pilot: fixed-prompt response comparison, fixed-versus-free
condition contrast, and an exploratory association with measured within-session
mood change. It is **not** evidence for a stable personality trait, a
cross-context response operator, or the V6 two-epoch-by-two-replica dynamic
object.

## Data handling boundary

The importer reads private text only in memory. Its outputs contain aggregate
counts or pseudonymous metadata; no source participant IDs, user text, or AI
text are written by this script. AI turns are never treated as participant
responses. They remain turn/character-count opportunity covariates.

The text exporter is resolved through `decrypted_output/` when the complete
experiment bundle is supplied. Adjacent questionnaire tables are registered as
withheld references only: they are neither loaded into this audit nor used for
SUICA training or external-validity claims.

## Imported text views

{view_summary.to_markdown(index=False, floatfmt='.2f')}

## Direct-answer recovery

The Markdown export supplied all three direct MEPS answer blocks. The fixed
Japanese rhetorical scaffold is removed before future scoring and retained only
as a slot indicator (`first`, `then`, `next`). `reported_answer_chars` is an
export-side field that includes scaffold/formatting conventions, so it is an
integrity check rather than a text-score denominator.

{direct_summary.to_markdown(index=False, floatfmt='.2f')}

## Mood-state coverage

{mood_summary.to_markdown(index=False, floatfmt='.3f')}

## What this corpus can test now

1. **Condition-fixed response profile:** model a score as `score[u, prompt] =
   prompt effect + author-within-protocol deviation + residual`. This is a
   conditional text-behavior comparison, not a personality trait score.
2. **Free versus fixed response:** compare the same person's task-answer view
   with their free-chat view while retaining text volume, turn count, and AI
   exposure as opportunity variables.
3. **Exploratory state linkage:** after freezing a Japanese representation and
   score set, relate free-chat descriptors to the pre/mid/post mood trajectory;
   do not train the descriptor on mood labels.

## What must wait for a new design

- Two or more sessions/days per participant;
- at least two independent replicas of each common condition per epoch;
- a frozen Japanese or multilingual embedding/segmentation layer and its
  language-specific reliability audit;
- a preregistered external criterion if any trait or clinical claim is wanted.

## Current protocol facts

- Participants discovered: `{metadata['n_discovered_participants']}`
- Participants loaded: `{metadata['n_loaded_participants']}`
- Missing source files: `{len(metadata['missing_files'])}`
- The correct analysis unit is **user-authored text in its declared condition**.
  Assistant language is context, never a participant score input.
"""


def main() -> None:
    args = parse_args()
    views, mood, metadata = load_meps_ai_conv(args.input_root, include_mood=args.include_mood)
    qc = metadata.pop("direct_answer_qc")
    layout = resolve_meps_experiment_layout(args.input_root)
    references = reference_registry(layout)
    view_summary = _view_summary(views)
    mood_summary = _mood_summary(mood)
    direct_summary = _direct_answer_summary(qc)
    readiness = _readiness(views, mood, qc, metadata)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    public_metadata(views).to_csv(args.output_dir / "view_metadata.csv", index=False)
    view_summary.to_csv(args.output_dir / "view_coverage.csv", index=False)
    mood_summary.to_csv(args.output_dir / "mood_coverage.csv", index=False)
    direct_summary.to_csv(args.output_dir / "direct_answer_recovery.csv", index=False)
    references.to_csv(args.output_dir / "reference_registry.csv", index=False)
    (args.output_dir / "layout_manifest.json").write_text(
        json.dumps(metadata["layout_manifest"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "data_schema.json").write_text(
        json.dumps(metadata["source_schema"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "readiness_decision.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        _report(view_summary, mood_summary, direct_summary, readiness, metadata),
        encoding="utf-8",
    )
    print(f"verdict={readiness['verdict']}")
    print(f"report={args.report}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
