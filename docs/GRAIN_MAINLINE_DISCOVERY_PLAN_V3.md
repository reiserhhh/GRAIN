# GRAIN mainline discovery V3

2026-09-07. User-directed return to discovery after the synthetic P1R/S/T packet.

## One finite first experiment: D1 real-author query readout

Question: On the same real authors and text budget, which personality associations
are present in the frozen LLM's trait-query responses, and how do they relate to
the existing BGE/E5/Qwen statistical readouts?

- Use all locally available complete native author records from the existing
  frozen direct collector. If the completed remote cache is available, copy it
  without restarting collection. Commit a membership/input snapshot before
  joining outcomes. Availability is not chosen from scores.
- Include Big5-only and MBTI-only source users through the existing reference-
  excluded OOF packets. Do not read bridge labels, MEPS questionnaires, or lockboxes.
- Retain raw, cue-mask and placebo-mask arms and all traits. Native score is the
  original equal mean of three views with the fixed answer direction. Do not flip
  axes, tune thresholds, calibrate on these labels, or choose favorable users.
- Report Pearson/Spearman for each Big5 trait; AUC, balanced accuracy and score
  spread for MBTI. Compare each on identical users with all three existing heads.
  Summarize query-to-head correlations and author-label linkage as two separate
  observable relations. No additional model fitting or inference in D1.
- Publish aggregates and local private artifacts with provenance. A discovery
  snapshot is not the old full128-shard benchmark: leave that entry point and its
  completion contract unchanged. These exposed users are discovery material for
  future tuning, not an untouched confirmation set.

Budget: one cache-analysis packet, no GPU, no API, no new training. Basic join,
direction, complete-view and metric tests suffice. Do not add a new defense series.

## Follow the signal

Use D1 to choose a concrete trait/evidence pathway for D2: query-first evidence
selection and representation intervention on real text, using new authors for
the resulting hypothesis. Then D3 tests whether independent GRAIN author
neighborhoods organize those pathways. D4 fits a compact response approximation.
The endpoint is an observable real-author evidence -> vector -> trait-judgment
chain; exhaustive prompt invariance is not required to start D2.

Every packet ends with a finding and one mainline next action. If a selected
observer gives no useful relation, move to a suitable existing observer/material
or report the specific resource needed; do not keep polishing its answer codes.

## Execution status

D1 is COMPLETE. The47 completed Windows shards provided213 eligible Big5-only
and2,328 MBTI-only authors. Native Big5 mean r=.1993 versus paired E5 .1372;
native TF AUC=.6600 and query/E5 score r=.5127. See
`../reports/grain_vector_mechanisms/D1_REAL_AUTHOR_DISCOVERY_REPORT.md`.

D2 now prioritizes TF's shared route and neuroticism's different-access route
(native r=.2244 versus E5 .0417 on this Big5 subset). Use bounded real-author
query-conditioned states and selected-evidence interventions, followed by GRAIN
neighborhood pooling. Old queues and P1U remain parked; GRAIN's released
deterministic scorer is unchanged.

D2 is now RUNNING:64 new source authors,128 two-view units,1156-forward budget.
See `GRAIN_REAL_AUTHOR_PATH_D2_PLAN.md` and `GRAIN_QUERY_PATH_MATHEMATICS_V1.md`.
The same frozen packet feeds D3's independent-neighborhood comparison and D4's
held-author rank2/rank8 response operators; their analysis script is implemented
and tested while neural collection proceeds. No source-label outcomes are read
in D2-D4.
