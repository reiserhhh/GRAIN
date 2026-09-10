# NEXT48: one source-fixed query-difference correction, then close MLP24 expansion

2026-09-10. Planned, not executed. NEXT47's fixed common reader remains useful.
The new question is whether the omitted current query-operator term has a
transferable source-common query component, rather than an indispensable
current-author gate dependency. Do not turn this into a residual-fit search.

## Fixed candidate

Use the exact NEXT47 source bank, current d/shared w*,12 authors,two queries,
four GRAIN directions and both modes. Source4 leave-one-author-out; new8 all4.
Compile independently for each query:

    v_q = mean_(s in allowed sources) J24(x_s,q)^T w*
    v_common = (v_N+v_C)/2
    v_difference = v_N-v_C
    source_query_term = v_difference^T (d_N+d_C)/2
    predicted_contrast = v_common^T(d_N-d_C) + source_query_term.

v_common must equal NEXT47's frozen v within1e-10. Direct same-source local
operator application must equal the compiled actions within1e-10. No learned
coefficient, blending, centering selected by targets, or source expansion.

Current prediction reads only the compiled vectors and d. Compilation reads
only source4 x/weights, with new8 x, native target increments, native gradients,
and previously evaluated o blocked by a data-file guard. Freeze all predictions
before evaluation opens current carried/operator/native endpoints.

## Mechanism outputs

- Query correction vs NEXT46 observed o: r, RMS, SSE/zero, signs; all cohorts,
  modes and individual factors. This directly tests the missing term rather
  than only final native-action correlation.
- Corrected contrast vs native: same metrics and signed final-T projection,
  alongside the unchanged NEXT47 common candidate.
- Exact error split: source-common minus current carried, plus predicted-query
  minus current operator term, plus saved floating-point parity. Keep covariance.
- If correction reduces operator-target SSE below zero baseline in both new8
  modes, retain a source-query candidate; if not, keep the common reader and
  record current context variation as the remaining local dependency. A useful
  correction must also report whether total contrast error improves or worsens.
  Do not hide a cancellation that makes total score look better.

## Resource / stopping rule

0 neural/scoring/backward/API/fit/human-label reads.60s numerical cap, one exact
array replay, no new capture. Same-model/same-query source transfer only, not
human personality validation or target-semantics isolation.

This is the last MLP24-only refinement in the current sequence. Regardless of
outcome, update the full dependency map and return to a bounded upstream-input/
transport integration decision. Do not add another alpha, layer, factor or
query search to pursue r=1. Keep useful approximate readers.

Proposed `scripts/run_grain_source_query_correction.py`, private
`logic_source_query_correction_v1/`, aggregate report/table, source and target
access audit, tests, local commit. Releases/scorer/lockbox and MEPS unchanged.
