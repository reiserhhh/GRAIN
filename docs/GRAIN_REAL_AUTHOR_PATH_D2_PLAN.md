# D2: real-author evidence and vector pathway discovery

2026-09-08. Implements the discovery-first mainline after D1.

## Question and finite packet

Which real-text evidence is reached by a personality query, and how much of its
effect travels through a query-end contextual vector? Start with TF (shared
native/statistical signal) and neuroticism (different native/statistical access).

Select40 MBTI-only and24 Big5-only authors by a fixed label-free hash, excluding
all D1 cached authors and reference users. Require existing all-arm GRAIN profile
support. Use the first two existing12-comment views and the cue-mask arm, which
already has usable D1 signals. No bridge, questionnaire, or new human label read.
Freeze half of each task's authors for discovery and half for subsequent response
prediction, without looking at model responses. This split supports D4, not a
relabeling of previously published source scores as a fresh validation study.

## Query-guided evidence selection

Use the cached E5 comment vectors and four newly encoded fixed trait descriptions.
With high/low description vectors q+ and q-, rank each view's comments by

    relevance_j = abs(cos(e_j, q+) - cos(e_j, q-)).

Keep the four largest values, retaining both poles of a trait. The comparator is
four other comments with the nearest total Qwen token length, deterministic tie
breaking. There is no label- or LLM-output-tuned selection. This is a concrete
query-first discovery operator, not a new universal retrieval theorem.

## Five natural inputs and four vector interventions

For each author/view, score original12 comments, remove-selected8, remove-control8,
selected-only4, control-only4 with the existing frozen native trait query. Preserve
within-view order. Capture block16 query-end and final-normalized query-end states.

For each removal donor h_d and original recipient h_0, intervene at block16:

    h(alpha) = h_0 + alpha * (h_d - h_0), alpha in {.5, 1}.

Keep all other recipient states/input unchanged. Save conditional trait score,
candidate logits, vector displacement, and the full intervention response. The
natural evidence effect and the transmitted single-position effect are separate
measurable quantities. A partial transmitted effect is useful progress.

One original repeat and self-patch per trait check the instrument. No all-layer,
answer-code, or prompt sweep. Fixed Qwen2.5-1.5B snapshot, eager FP32 on local MPS;
last-token vocabulary readout avoids allocating logits at every text token.

## Resources, analysis and next links

-64 authors x2 views x9 forwards=1152 forwards, plus4 basic instrument checks.
-One E5 anchor batch, then unload E5 before loading Qwen. No downloads or API.
-Max neural wall time30 minutes; preserve completed author units on interruption.
  Evaluate feasibility on the first complete author, not on outcome significance.
-D2 reports query-guided versus matched removal effects, selected-only versus
  control-only agreement with the full score, and vector-intervention transfer.
-D3 uses the frozen48-dimensional GRAIN whitened profile and16 identified landmark
  distances already available, not outcome-fitted groups, to compare response
  neighborhoods. Treat these as specified GRAIN channels, not the entire framework.
-D4 uses the frozen author split to test a compact direction/gain readout on new
  authors. Follow detected effects rather than perfecting a generic instrument.

All texts/author IDs/arrays remain in ignored results/. Publish aggregate results
and formulas. Retain all fixed conditions. Source labels are unnecessary for this
mechanism packet because D1 supplies the external association anchor.
