# NEXT51: source-selected evidence has a recoverable endpoint effect

2026-09-10. Complete after primary freeze `eb7ee35`. Same12 previously inspected
authors, N/confirmation queries, four frozen GRAIN BODY directions. Source4
leave-one-out; new8 uses all4 sources. No human labels or model fitting.

## Main finding

**The source-query selection component remains useful at the final logit
readout, not only under NEXT50's local projection.** Its same-query source-only
program gives new8 endpoint-component r=.972986/.978394 (N/confirmation),
SSE/zero=.156129/.071926. Both predeclared retention conditions pass.

| new8, same-query anchor | Actual suffix reader: diagnostic | Source-fixed suffix reader: program |
|---|---:|---:|
| N component r | .962138 | .972986 |
| N SSE / omission | .258700 | .156129 |
| Confirmation component r | .967181 | .978394 |
| Confirmation SSE / omission | .091643 | .071926 |
| N-minus-confirmation component r | .957183 | .963235 |
| Contrast SSE / omission | .367583 | .208588 |
| Predicted contrast RMS | .061323 | .085793 |
| Actual contrast RMS | .145007 | .145007 |

The compiled program retains59.2% of the actual contrast RMS and28/32 contrast
signs. Source4 compiled N/confirmation r=.968469/.875324, contrast=.921224.
New8 per-factor compiled contrast r=.981/.995/.976/.995. Source4 confirmation
F01=.739 and F02=.817; complete less favorable outcomes are retained.

The target is **one final-position attention22 key contribution to L's local
derivative**. It is neither all attention positions, the complete score L,
the older probability score F, nor a Big5/MBTI questionnaire prediction.
Each new8 aggregate pools8 authors x4 directions, not32 independent people.

## Query specificity changes with the readout

| new8 anchor | NEXT50 shared local reader contrast r | NEXT51 actual endpoint transport r | NEXT51 compiled endpoint r |
|---|---:|---:|---:|
| Same query | .939345 | .957183 | .963235 |
| Other query | -.890468 | .868015 | .446672 |
| Shared query | .477162 | .963435 | .921918 |

The local sign reversal under exchanged anchors **does not persist at the
endpoint**. NEXT50 remains correct for its fixed shared w*. At the endpoint,
each query has a different suffix reader and query-conditioned current evidence.
Their composition changes what a local vector difference means. Thus the query
anchor participates in selection but is not the sole location of query meaning.
Same-query remains the primary fixed candidate; no post-outcome arm/scale mixture.

## Error interaction and actual relevance to the full response

Let u be the native key vector, uhat the source-anchor vector, r the current
suffix reader and rbar the source reader. The following account is exact:

    compiled - actual = r^T(uhat-u) + (rbar-r)^T uhat
                      = selection_error + reader_error.

For new8 same-query contrast, the mean-squared terms are .00772920 selection,
.00077699 reader, and **-.00412020 cross-term**, totaling .00438599.
The fixed reader partly compensates the selection error here. That explains
why it can have lower output error than actual-reader transport of the same
approximate vector; it does not mean the actual derivative is wrong or that
the two errors are independent. Source4 contrast instead has a positive cross-term.

The true component's signed projection onto the full H16 derivative tau is
.251619 for new8 N, +.094271 for confirmation, and+.276996 for their contrast.
These are response projections, not percentages of personality information.
Source4 contrast projection is+.058165. Do not rank psychological importance
from these source/cohort differences.

Replacing just this component while borrowing all other native contributions
gives new8 contrast correlation .993669 with actual-reader transport, .995208
with the compiled component. These high numbers are **replacement audits**,
not standalone prediction accuracy: most of tau is still supplied by the native
reference. The source-only component's .963235 and its amplitude/error are the
appropriate headline. Local coordinate fidelity and whole-function fidelity
are different estimands, not competing truths.

## What was executed

1. Construct all24 full candidate vector packets from existing K/V/dK, source Q
   and actual W_O, before any suffix targets. No current Q/readout in construction.
2. Capture8 source-query suffix gradients, then compile same-query rbar with
   source exclusions and freeze all12 scalar prediction packets.
3. Only then capture16 new-query suffix gradients. Same original full-sequence
   suffix, norm/head shape and fixed logit contrast c; no prefix recomputation.
4. Evaluate actual u using current Q, and tau from the full pre22 gradient
   contracted with its transported GRAIN direction. Current gradients are
   evaluation-only; the compiled component never reads them.

Native work:24 suffix forward/reverse pairs,2 JVP checks,4 central-difference
forwards, **20.364872s** total compute. Source capture7.643056s; aggregate loading
5.254325s, capture wall21.231172s. No live run, API, fitting or human-label reads.
Preparation.859685s; vector construction.747826s, scalar prediction.041918s;
cache-only replay.753499s and.042949s. Numerical stages make no neural calls.

## Verification and reproduction

-24 vector packets and12 scalar prediction packets replay identically;7 CSVs
  and summaries agree.288 component actions/144 contrasts, exact12-author inventory.
- Actual-Q full key vectors match NEXT49 within2.110e-15. Projection onto both
  prior local readers recovers NEXT50 within3.331e-16.
- Final L agrees exactly with NEXT39 on all24 inputs.96 full H16-chain actions
  agree with the original end-to-end gradients within4.777e-6; two new JVP/
  reverse checks differ by at most7.616e-7.
- Two post-attention finite-difference checks meet the frozen tolerance;
  largest absolute difference1.314e-4. Independent summary difference2.887e-15;
  squared-error account closes within1.195e-16. All8 new authors use the same
  source reader per query. Both construction stages pass four denial probes.
-67 related tests pass,18 existing PyTorch/Python3.14 warnings. Ruff and diff
  checks pass. An audit-only initial H16 read assumed a batch axis; corrected
  to the existing token-by-hidden cache schema before the audit passed. Frozen
  primary computation, predictions, thresholds and native outcomes unchanged.

Private packet: `results/grain_vector_mechanisms/logic_selection_endpoint_v1/`.
Public aggregates: [query](selection_endpoint_tables/query_summary.csv),
[factor](selection_endpoint_tables/factor_summary.csv),
[contrast](selection_endpoint_tables/contrast_summary.csv),
[factor contrast](selection_endpoint_tables/contrast_factors.csv),
[error account](selection_endpoint_tables/error_accounting.csv).

```bash
PY="../project persona/.venv/bin/python"
# Completed packet: audit only; do not restart native capture.
$PY scripts/audit_grain_selection_endpoint.py
# Fresh private packet, staged in exactly this order:
$PY scripts/run_grain_selection_endpoint.py prepare --output /private/new_packet
$PY scripts/run_grain_selection_endpoint.py vectors --output /private/new_packet
$PY scripts/run_grain_selection_endpoint.py capture-source --output /private/new_packet
$PY scripts/run_grain_selection_endpoint.py predict --output /private/new_packet
$PY scripts/run_grain_selection_endpoint.py capture-new --output /private/new_packet
$PY scripts/run_grain_selection_endpoint.py evaluate --output /private/new_packet
```

## Next mainline decision

The source selector and suffix reader are now a useful component pair; stop
refining this location. The remaining runtime dependence is current K/V/dK,
produced by native blocks17–21. NEXT52 tests a single differentiable source-only
state-transport map from cached H16, with a pointwise/causal-context comparison.
The same map must provide both its state output and its directional derivative;
do not fit unrelated arrays and pretend they describe one function.

See [theory](../../docs/GRAIN_SELECTION_ENDPOINT_THEORY_V1.md) and
[NEXT52](../../docs/GRAIN_SOURCE_STATE_TRANSPORT_PLAN_V1.md). The broader goal
remains ACTIVE. No release/scorer/lockbox/MEPS changes or remote push.
