# Source endpoint assembly: a current-reader-free local query function

2026-09-11. NEXT77 / finite pass66. COMPLETE, audit PASS. Primary implementation
`0fc049d`; fixed [plan](../../docs/GRAIN_SOURCE_ENDPOINT_ASSEMBLY_PLAN_V1.md).

## 1. What this adds

The existing source-only endpoint reader can be composed with both numerical
selector/value functions. Application now uses H16/layout/fixed weights/source
maps only: no current Q/K/V, no material-specific g, no current endpoint gradient,
no native remainder add-back. It produces a local scalar and its compatible
GRAIN-direction derivative, not just an action assembled from unrelated pieces.

There is useful query-directed information: new8 query-specific scalar contrast
r ranges .545-.690. Its response contrast also correlates .671-.745, but is too
large. The experiment identifies a dependency that can be removed and an error
interaction that must be modeled. It does not promote a whole-personality scorer.
Both source4LOO and new8/all4, both queries, all four factors and all four
predeclared functions are retained; no lower-error arm was selected afterward.

## 2. Exactly what is being compared

For query q, source reader rbar_q, output matrix O and selected content v:

    Fhat_q(H)=rbar_q^T O concat_h sum_t a_hat_qht(H) Vhat_t,kv(h)(H)
    dFhat_q[D_f]=rbar_q^T (u_key_f+u_value_f).

The target uses cached native-Q K/V vectors and the corresponding *current*
endpoint reader r_q. Q and endpoint reader are fixed in this local derivative;
DQ, residual identity, MLP and other positions/layers are outside this component.
The scalar excludes output bias and is a local fixed-reader functional, not L.

Two endpoint modes: query-specific reader and paired-query mean reader. These
change the target, not only the prediction. They are not old own/pooled g modes;
do not compare their errors as if all measured the same variable. Oracle rows
use native K/V, never candidate application inputs. NEXT51 source-reader and
source-key predictions are independently reconstructed before summary.

## 3. New8 quantitative results

Query-specific N-minus-confirmation, n=8, four directions per author for actions:

| Selector | Values | Scalar r | Scalar SSE/zero | K+V r | K+V SSE/zero | K+V RMS |
|---|---|---:|---:|---:|---:|---:|
| identity | identity | .686424 | .436734 | .735597 | 21.250069 | .648428 |
| identity | source | .689817 | .225107 | .745173 | 20.127328 | .634545 |
| response coordinates | identity | .544930 | .342247 | .670569 | 7.092794 | .397206 |
| response coordinates | source | .577035 | .193610 | .677814 | 7.288066 | .402900 |
| source Q / native KV (oracle inputs) | native | .409900 | .082914 | .852352 | .416889 | .057131 |
| native Q / native KV (reader-only oracle) | native | .490437 | .098647 | .956158 | .891104 | .227159 |

Target scalar RMS=.654353; action RMS=.122288. SSE/zero compares with predicting
zero, not with an optimized population mean. All numerical scalar contrasts have
the same sign as the target for8/8 authors; this is not classification accuracy.
Scalar ordering, absolute magnitude and directional sensitivity are distinct.

The fully source-assembled response/source function gives per-query action
r=.841738/.836428 and SSE/zero=5.183960/3.995333 for N/confirmation. Thus the
contrast result is not evidence of accurate action amplitude for either query.

In query-pair-mean mode, the same function gives scalar contrast r=.028134,
SSE/zero=1.558088 and0/8 matching signs. Its K+V contrast r=.687120,
SSE/zero=2.081440 (target RMS=.141140). The large scalar difference between modes
motivates separating common-read and reader-difference terms, not dropping a mode
or choosing its smaller action error as a universal improvement.

Source4LOO supplies an important counterpart: response/source scalar contrast
r=-.778636 although SSE/zero=.212643, and K+V r=.684494/SSEzero=.554387. The
response/identity function gives K+V r=.750352/SSEzero=.443951. Successful new8
scalar ordering is therefore not a stable all-panel ordering claim.

## 4. The mechanism of composite error

For each actual vector u, approximation uhat, actual reader r and source reader
rbar, preserve the exact identity:

    e_transport = r^T(uhat-u)
    e_reader = (rbar-r)^T uhat
    e_total = e_transport+e_reader
    MSE_total = MSE_transport+MSE_reader+2 mean(e_transport*e_reader).

New8 query-specific K+V contrast, unnormalized squared response units:

| Function | Transport MSE | Reader MSE | Signed cross | Total MSE |
|---|---:|---:|---:|---:|
| response / identity value | .028774 | .046370 | +.030924 | .106069 |
| response / source value | .028029 | .046264 | +.034696 | .108989 |
| source Q / native KV | .011670 | .001558 | -.006994 | .006234 |
| native Q / native KV | 0 | .013326 | 0 | .013326 |

The source value fit lowers the first two squared terms separately in this
contrast, but their increased positive alignment worsens total error. Conversely,
the source-Q/native-KV row partly benefits from cancellation; its lower error
than native-Q does not establish a universally superior query anchor. The
account is exact but ordering-dependent, not unique causal variance allocation.

The global contrast vector error also improves for response/source values
(SSEzero1.351021->1.335084) while endpoint action error worsens. This extends
NEXT76's observable-geometry finding through the *actual query endpoint*.

## 5. Verification and artifacts

- 24 source-reader checks reproduce NEXT51/NEXT52 reader arrays exactly.
- 64 full-scalar central finite differences: max error2.720824e-10.
- Old key vector/action replay max4.884981e-15; reader identity3.552714e-15;
  squared-error identity1.110223e-16.
- 24 prediction packets and seven aggregate CSVs replay exactly.
- Successful numeric stages18.784439CPU seconds including replay; zero fits,
  neural calls, APIs, human labels or MEPS reads. H16 remains neural-derived.
- 34 focused synthetic/compatibility tests pass; Ruff/format/whitespace checks
  pass. Existing PyTorch/Python3.14 deprecation warnings only. An initial test
  invocation named a nonexistent guard test and collected none; the corrected
  final invocation above completed. No full-repository test claim.

Public tables: [publication inventory](source_endpoint_assembly_v1/publication.json),
[query actions](source_endpoint_assembly_v1/query_actions.csv),
[factor actions](source_endpoint_assembly_v1/factor_actions.csv),
[contrasts](source_endpoint_assembly_v1/contrast_actions.csv),
[scalars](source_endpoint_assembly_v1/scalar_metrics.csv),
[components](source_endpoint_assembly_v1/component_summary.csv),
[error account](source_endpoint_assembly_v1/reader_error_account.csv),
[vector metrics](source_endpoint_assembly_v1/vector_metrics.csv).
Private arrays remain under `results/grain_vector_mechanisms/logic_source_endpoint_assembly_v1/`.

## 6. Decision

Retain the source-only local function and the error-interaction mechanism. Close
this assembly card without a new parameter search or release change. Next ask
which part of the personality-query contrast is shared-reader evidence movement
and which part is query-reader contrast applied to common evidence. This uses the
now-composed endpoint, not a fresh V-accuracy or readout-alpha grid. A larger
encoder/complete endpoint development decision follows that bounded distinction.

The same-material GRAIN interface is dF_q[D_f], a signed and query-conditional
quantity; it is not simply a positive importance weight. Future relational or
psychodynamic observation queries can use this interface after defining their
evidence units and coding targets. No clinical interpretation is generated here.
