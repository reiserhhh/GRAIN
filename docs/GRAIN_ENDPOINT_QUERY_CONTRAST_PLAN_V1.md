# NEXT78: common-reader evidence and differential-reader query contrast

2026-09-11. PLANNED, not executed. One fixed-array mechanism distinction after
NEXT77; zero fits/neural/API/Y,30CPU seconds including replay. No parameter grid.

## Question and mainline decision

NEXT77 composes numerical selection/content with source-only endpoint reading.
Query-specific scalar contrast preserves some new8 ordering, whereas query-mean
scalar contrast has r near zero for the response selector. Which part carries the
query contrast: changing evidence under a shared reader, or a changing reader
applied to shared evidence? This determines whether the next development should
target differential reading or the upstream evidence representation.

The question concerns endpoint reading, not NEXT73's earlier Q-dependent
selection-vs-orientation experiment. Reuse the NEXT77 functions and targets; do
not refit a source mean, rescale an action, select a query or add another encoder.

## Frozen decomposition

For each author, both queries, all four numerical arms and both oracle-input
comparators from NEXT77, use the complete post-output-projection primal p_q and
key/value/total tangent vectors u_qf. Source readers obey the same old source4LOO
and new8all4 membership. Native current readers are evaluation-only.

    mean_r=(r_N+r_C)/2; diff_r=r_N-r_C
    mean_p=(p_N+p_C)/2; diff_p=p_N-p_C
    contrast = mean_r^T diff_p + diff_r^T mean_p
             = evidence_change + reader_difference.

Use the same identity for each u component and GRAIN factor. The first term must
exactly reproduce NEXT77's query-pair-mean contrast; the sum must reproduce its
query-specific contrast. Never compare the two modes as one unchanged target.

## Execution

1. Pin NEXT77 audit/publication/private prediction hashes. Reconstruct only the
   already cached evaluation targets with the same NEXT77 code; no new forward.
2. Export both terms and their signed sum for all authors privately. Publish
   source4/new8 aggregates: RMS, mean, covariance/cross energy, target fidelity,
   and per-factor counterparts. Do not treat four factors as independent people.
3. Decompose errors of both terms; keep the signed cross term. Compare native
   reader-only and source-Q/native-KV controls to distinguish readout cancellation
   from successful upstream transport. Source and target decompositions keep
   their own readers; no native remainder enters candidate prediction.
4. Use source-only reader span/common-difference geometry as descriptive context,
   not a new learned source basis or a claim of unique personality coordinates.
5. Check exact identities/replay and publish the complete two-query grid. No
   thresholds learned from new8, causal percentages or human-validity labels.

Suggested `endpoint_query_contrast.py`,
`scripts/run_grain_endpoint_query_contrast.py`, focused tests/report under the
usual vector_mechanisms paths. Close after one decomposition. If reader-difference
dominates useful contrast, next define a source query-conditioned reading
function; if evidence-change dominates, develop upstream contextual transport.
If cancellation is dominant, preserve their joint scalar instead of pruning a
term. A result not distinguishing these alternatives closes this account and
returns to a new full-endpoint/encoder mechanism, not another attribution grid.

The whole-system goal remains ACTIVE. H16 remains neural-derived, the measured
component excludes DQ/other layers, and no fresh real-author criterion is added.
