# NEXT44: shared compiled reader versus query-conditioned incoming direction

2026-09-10. Planned, not executed. NEXT43's paired-source block25 derivative
retains new8 action r=.996936/.998451 and compiles exactly into wbar^T d.
The next question concerns where query specificity resides, not a better fit.

## Decision / fixed comparisons

Use the same24 author-query units, N/confirmation, four GRAIN directions,
own/pooled source readouts. No additional native calls, gradients, labels or
source fitting. All current inputs are already frozen boundary tangents d;
wbar is source-only, with old4 leave-one-out and new8 fixed-four membership.

Verify same-author material/factor alignment from the saved H16 BODY/GRAIN
inputs. Keep the recipient's tangent d_q unchanged and evaluate the full2x2
matrix of compiled readers:

    tauhat_(q <- p) = wbar_p^T d_q.

Own-query diagonal must reproduce NEXT43. Off-diagonal borrowing uses the
corresponding source construction for the same held author, not their current
state or an average including that author. Keep both borrowing directions and
both t modes; no best-per-query mixture or coefficient rescaling.

Targets remain the recipient query's saved native block25 increment read by
its own frozen recipient t. Using a donor w is the tested approximation; do
not also change the target to the donor's t to improve its apparent accuracy.

## Exact query-contrast decomposition

On each same-author/direction pair, define

    contrast = w_N^T d_N - w_C^T d_C
    reader_part = (w_N-w_C)^T (d_N+d_C)/2
    transport_part = (w_N+w_C)^T (d_N-d_C)/2
    contrast = reader_part + transport_part.

This is an exact symmetric allocation of the compiled program, not a unique
causal mediation analysis. Report signed components, their cross-term, RMS,
and relation to the native recipient-action difference. If the contrast is
tiny, retain its absolute scale rather than interpreting an unstable ratio.
Model layer depth/query differences are not human temporal state changes.

Compare cosine and norm of the source w pair to the current d pair; ordinary
angles alone do not identify the complete transport Jacobians. No claim that
similar readers imply similar psychological constructs or a real scale bridge.

## Implementation and bounds

1. `compiled_gated_readout.py` already exists and is verified; reuse it without
   refitting. Add a small query-pair algebra helper and synthetic tests for
   asymmetric borrowing, signed cancellation and source exclusion.
2. Freeze input/code/design hashes; numeric prediction process may read only
   compiled w, metadata and current d. Deny native increment/analysis targets.
3. At most384 donor-recipient/factor/mode predictions on24 units, plus exact
   paired diagnostics after freeze; use no native model, checkpoint or raw texts.
4. Primary metrics: recipient action r, factor-centered r, SSE/zero, RMSE,
   amplitude and signs, all four factors, old4/new8 separately. Own diagonals
   replay<=1e-12. Target action/contrast addresses stay explicit.
5. Capture full covariance/RMS accounting for reader/transport contrast parts.
   No p-value screen or tuning. Numerical cap60s; no unattended alternative
   model, rank search or source expansion. One numerical replay suffices.

## Finish / branch

If reciprocal borrowing retains the own-reader effect, keep a shared local
reader and next focus on the query-conditioned upstream tangent. If borrowing
has a material loss, preserve that query-specific reader component and
localize its source operator/output-direction dependence. Contrast accounting
can identify which difference is being preserved, not just whether r is high.

Either result must yield the next concrete input/transport mechanism question.
Do not extend this into an all-layer grid solely to produce a larger table.
General raw-text extraction, unseen material and criterion links remain later
integration requirements, not reasons to erase the present usable local map.

## Deliverables

Proposed `run_grain_compiled_query_transfer.py`, synthetic tests, private
`logic_compiled_query_transfer_v1/`, aggregate `COMPILED_QUERY_TRANSFER_REPORT_V1.md`,
updated dependency/logic ledger and local commits. Preserve old release/scorer,
lockbox, MEPS boundary and interpretation protocols.
