# HM12: Mean Displacement Versus Within-Span Organization

2026-10-08. Not executed. Finite result-guided continuation after HM11,
reusing controlled HM10B3 materials; not new independent natural text data.
No change to any frozen primary/helper/config, no rank/alpha/accuracy grid.

## Decision

HM11's source projection is not a sufficient standalone observation. Before
learning a larger representation, distinguish two losses: vector components
outside U, and token differences discarded by span averaging. If averaging
itself hides target action, a larger frame of means is the wrong next design.

## Fixed Materials and Reader

- Same 128 same-property other-to-self pairs from HM10B3; roles remain
  source/validation/application 8/4/4 lexical families.
- All three unchanged queries, original block output16 only. No best-layer
  choice and no new persona generation, APIs, private text or neural fitting.
- Preserve fixed receiver input and original block16 question states while
  patching only the property span. Recompute all later blocks. This is a
  local total perturbation, not a natural paraphrase or storage-only mediation.
- Verify model/config/tokenizer hashes and the existing raw span alignment.
  Existing full/property and unpatched outputs are references, not new calls.

## New Arms Before Calls

For token displacement d=donor-current at the same property span,
bar(d) is its token mean and r=d-bar(d). Two new actual arms:

1. mean_only: current_j + bar(d) for each property token j.
2. centered_only: current_j + r_j for each property token j.

These sum to the original full displacement before float32 rounding. In the
ideal coordinates, centered_only preserves every full-dimensional span mean,
not just U coordinates. Reconstruct the actual rounded mean shift too.
One-token spans make centered_only exactly a no-op; run and report them as
an internal control, not missing observations. Keep multi-token and one-token
results separate rather than letting their differing opportunity mix drive
a headline. No prompt or pair filtering after seeing output differences.

## Bound and Checkpoint

768 new calls =128 pairs x3 queries x2 arms. Enumerate all addresses and
freeze the new helper/config/plan before the first call. Use resumable atomic
units; at most400 calls or900sec per stage. No duplicate collector and no
concurrent evaluation. Track actual calls/time, model replay, span live-state
error and numerical rounding. Do not replay old9696 interventions.

## Main Comparisons

- Per pair/query: mean-only and centered-only signed action versus unchanged
  receiver. Report full = mean + centered + finite mixed response; no additive
  assumption or percentage of personality information.
- Same-word owner-by-polarity contrasts for each axis/order/query, all roles.
- Full versus mean-only, centered-only versus baseline: whether equal or
  near-equal full-dimensional means retain different downstream observations.
- Conditional empirical family intervals; source frame/neural uncertainty
  and human-population uncertainty are not included.
- Single-token no-op max error and all hash/arithmetic checks.

If centered-only carries action with negligible mean change, retain token
organization in the next observation unit (sequence/second moments or typed
token relations). If mean-only reproduces the full action and centered-only
is inert, expand source-supported vector directions while retaining event
binding. If mixed response dominates, learn the conditional joint function,
not independent scalar feature weights. None implies a human trait by itself.

## Deliverables and Scope

New runner/helper/config/tests, frozen addressed interventions, mean-change
and mixed-action tables, source/validation/application results, independent
arithmetic verification and mechanism report. Preserve HM11 functions and
its fiber diagnostic, including failures. Do not silently relabel this plan
as a pre-registration of already inspected output.

Natural author aggregation, independent eligible evidence, individual B_u,
psychological interpretation, transport and the original18/100/44 remain
open. This experiment resolves a representation design decision in that
chain, not the whole theory or an accuracy benchmark.
