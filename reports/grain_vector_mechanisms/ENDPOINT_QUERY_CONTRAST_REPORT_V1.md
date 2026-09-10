# Query contrast: shared evidence, differential reading and author variation

2026-09-11. NEXT78 / pass67, COMPLETE, audited. Primary `241fd56`;
[fixed plan](../../docs/GRAIN_ENDPOINT_QUERY_CONTRAST_PLAN_V1.md).

## 1. Finding and development decision

The two queries use strongly overlapping source endpoint directions, yet their
small differential reader is important for scalar contrast. More importantly,
that term can *oppose* rather than amplify GRAIN-direction response. Replacing
current readers with a constant source reader loses this modulation on new8.

Therefore develop one **context-conditioned differential reader** next. Preserve
the shared reader and all existing selector/value functions. Do not restart a
generic embedding/value-accuracy grid. The next function should vary its readout
across author contexts while keeping the same exact scalar/body derivative.

## 2. What was calculated

For N=neuroticism and C=confirmation_seeking (not conscientiousness), let p_q be
the post-output-projection attention component and r_q its fixed endpoint reader.

    contrast = r_N^T p_N - r_C^T p_C
             = mean(r)^T (p_N-p_C) + (r_N-r_C)^T mean(p)
             = evidence_change + reader_difference.

The same formula applies to key/value/total directional vectors. All four NEXT77
functions and two native-input oracle comparators remain. Source4LOO and new8all4
are separate reused discovery panels. Current native readers enter evaluation
only. No model fitting, new neural/API calls, Y, MEPS or raw-text rereading.

Raw energy and author-centered covariance are both reported. For action columns,
centering is within each fixed GRAIN factor across authors before combining
columns. Four factors are not four extra people. Centering is descriptive, never
a new calibrated prediction or a rule to remove shared evidence.

## 3. Native target: scalar and response use the two routes differently

The projections below mean Cov(term,total)/Var(total) after author centering.
They sum to1 by algebra, can be negative or exceed1, and are not causal shares.

| Panel | Observable | Evidence projection | Reader projection | Signed cross covariance |
|---|---|---:|---:|---:|
| source4 | local scalar | .449683 | .550317 | +.012227 |
| new8 | local scalar | .259302 | .740698 | +.002164 |
| source4 | K+V direction response | .560223 | .439777 | +.000604 |
| new8 | K+V direction response | 1.128743 | -.128743 | -.005405 |

Thus differential reading contributes appreciably to scalar author variation in
both panels. The opposing directional modulation appears on new8, not source4;
it is not asserted as a universal personality law. It reveals a computational
behavior which the current constant-reader replacement fails to preserve.

New8 native scalar means: evidence=-.182779, reader=-.440027, total=-.622805.
The differential reader is not merely a large constant offset: its centered
projection is .740698. However, the absolute score also contains a large common
offset, so low SSE against zero alone is not recovery of author distinctions.

## 4. What the source-assembled program retains and misses

For the frozen response-coordinate/source-value function on new8:

| Observable | Term | Target fidelity r | SSE/zero | Target RMS | Candidate RMS |
|---|---|---:|---:|---:|---:|
| scalar | evidence change | .028134 | 1.558088 | .206818 | .064695 |
| scalar | reader difference | .458460 | .112244 | .471577 | .449333 |
| scalar | total | .577035 | .193610 | .654353 | .397798 |
| K+V response | evidence change | .687120 | 2.081440 | .141140 | .276426 |
| K+V response | reader difference | -.344782 | 21.117165 | .035833 | .148014 |
| K+V response | total | .677814 | 7.288066 | .122288 | .402900 |

The shared-reader evidence term recovers some response ordering. The differential
reader term is oversized and negatively aligned to its native target, so it
amplifies instead of supplying the target's opposing modulation. Candidate
centered K+V projections .670497/.329503 contrast with native1.128743/-.128743.

The reader-only oracle isolates part of this gap: with native Q/K/V fixed,
substituting source readers gives differential-response r=.172628 and
SSEzero2.161903; the target's negative centered reader projection becomes
positive .122543. Upstream evidence need not be wrong for this loss to occur.
The source-Q/native-KV comparator gives r=.402778/SSEzero1.073133 for the
differential term; its useful total still includes cancellation. Neither oracle
is an independent deployable prediction candidate.

Scalar error is different: for the same numerical function, raw evidence/reader
term MSE=.066645/.024961, but author-centered MSE=.010404/.024894. Much of the
evidence error is a mean offset; variation error is larger in differential
reading. The correct development target is a context-dependent readout, not
just a constant correction to make raw SSE look better.

All four numerical arms and per-factor results remain in the published tables.
Source4 scalar behavior remains different: response/source centered projections
-.626109/1.626109, reflecting cancellation and the earlier negative total ranking.

## 5. Reader geometry and scope

The all4 source readers have cosine .955340. Their mean norm is1.179347,
difference norm .441109, and second/first singular ratio .147340 (rank2).
Source3LOO cosine averages .954123. Nearly parallel readers can still yield
substantial contrast after contraction with evidence; a small angle is not an
irrelevance test. These are endpoint directions, not discovered human trait axes.

The scalar uses a fixed internal origin. A common translation p_q->p_q+c leaves
the evidence-change term unchanged but adds (r_N-r_C)^T c to the reader term.
Compensating intercepts restore the same scalar; arbitrary recentering without
them changes it. This is why it remains a declared local functional, not the
model's complete L or a new standalone scale. An additional algebra test covers
this reference issue without changing any empirical outputs.

## 6. Verification and delivery

-12 candidate packets and12 private evaluation packets, five aggregate tables.
- Exact old-mode identity max3.108624e-15;288 old scalar/action metric values
  reproduced within4.690692e-14; raw/centered energy identities8.326673e-17.
- Primary and replay arrays equal; all five aggregate CSVs replay byte-for-byte.
-4.273223successful numeric-stageCPU seconds including replay; no fits/neural/Y.
-40focused tests plus2post-account reference/conditional-derivative tests pass;
  formatting/Ruff/whitespace checks pass. Existing PyTorch/Python3.14 warnings
  only; no full-repository test claim. Primary empirical code remained frozen.

Tables: [term fidelity](endpoint_query_contrast_v1/term_metrics.csv),
[factor fidelity](endpoint_query_contrast_v1/factor_metrics.csv),
[joint account](endpoint_query_contrast_v1/joint_account.csv),
[errors](endpoint_query_contrast_v1/error_account.csv),
[source geometry](endpoint_query_contrast_v1/source_reader_geometry.csv),
[provenance](endpoint_query_contrast_v1/publication.json).
Private artifacts: `results/grain_vector_mechanisms/logic_endpoint_query_contrast_v1/`.

Next: [context-conditioned differential reader](../../docs/GRAIN_CONTEXT_DIFFERENTIAL_READER_PLAN_V1.md).
The overall goal stays ACTIVE; this finite decomposition is complete.
