# Logic NEXT02: query-readable residual observations

2026-09-10, staged after the completed natural-transport packet. Exploratory,
not selected as a new blind validation. Decisions use the first packet: native vs
fixed-function deletion changes correlate .886 (T/F) and .681 (neuroticism),
whereas source GRAIN maps do not explain the same changes comparably well.

## Fixed follow-up, no new model calls or outcome fitting

Retain the same 276 source authors, twelve held authors, two queries and three
source-only map estimands. Fit each of the already frozen query-function parts
(mean, token-centered configuration, final readout state) using the same GRAIN
features, scaling and alpha=100. Their sum must equal the previously fitted
total-function map, apart from the fixed anchor/intercept. This is a linearity
test of the implementation, not a fresh model-selection stage.

For query q and part c:

    R_qc(text) = f_qc(H(text,q)) - [b_qc + beta_qc' G(text)].

Use the between-author map for positions and the transition map for changes;
retain within and between alternatives in the change table. The reference is
source authors, never the current scoring batch. For level scores, average source
residuals per author, freeze their center and standard deviation, then standardize
held full-text residuals. No human labels enter the score or its normalization.

For the existing E1 material change:

    Delta F = Delta GRAIN + sum_c Delta R_qc + (Delta F - Delta f).

Report all three residual parts separately, correlations with native changes,
RMS and signed contributions mean(C_j Delta F)/mean(Delta F^2). These sum to one
only with all represented/residual/reader-gap terms included. Negative and >1
values are legitimate cancellation, not percentages of psychological variance.

Compare matched and mismatched queries on the same twelve authors without
fitting a new projection. This begins NEXT04's query-selectivity question. Do not
infer global uniqueness from only two queries; selectivity and shared use may
coexist. Do not name components as psychological constructs from model targets.

## What this develops

A reproducible candidate measurement channel beyond the historical factor bank,
with a fixed reference and explicit query/encoder dependencies. It requires
frozen model states; it does NOT solve standalone raw-text-to-state extraction.
The scalar components have their own fixed mathematical meanings. A future
non-LLM readout or encoder can target these objects instead of hoping one
query-independent factor will represent every personality judgment.

Full/E1 changes are researcher-deleted available evidence, not spontaneous
personality change. E2 has no matching full P/Q cache and remains a native/semantic
control from NEXT01; no invented residuals for that arm.

The mean/configuration/final decomposition uses the existing fixed affine
query function. Linear attribution completeness is an accounting idea also
central to [Sundararajan et al. (2017)](https://proceedings.mlr.press/v70/sundararajan17a.html);
this experiment does not equate the affine decomposition to integrated gradients
of the nonlinear native model. The residual-inclusive path derivation and its
conditions remain in `GRAIN_LOGIC_SEMINAR_20260910.md`.

Private output: `logic_query_residual_v1/`; aggregate public report integrated
with `NATURAL_TRANSPORT_LOGIC_REPORT_V1.md`. Finish after all fixed parts and
queries are reported, tests pass and new next-card decisions are documented.
