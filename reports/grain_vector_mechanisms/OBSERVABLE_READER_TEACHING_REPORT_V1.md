# Source-observable reader teaching: NEXT83

2026-09-11. COMPLETE, finite logic pass72. Primary `6451b8e`. Overall goal ACTIVE.

## Finding and decision

Functional teaching was implemented and tested, not just proposed. It gives a
small further improvement in the fixed assembled scalar error, **0.58%**, but
does not improve the anchored K+V response. Preserve NEXT82's useful same-material
interface; do not continue loss-weight tuning on this four-source panel.

The new geometric result is actionable: the fitted readers are restricted to
27 source target directions. **42.78%-42.83% of recipient reader-vector error**
is outside that range. After evidence projection the inside and outside parts
can cancel or reinforce, so this fraction is not a functional-error share.
A mathematical witness also shows that adding functional losses can require
output directions outside the target span. NEXT84 will include the already
available source evidence directions, not arbitrary extra latent dimensions.

These are eight reused discovery authors, four teaching authors and fixed-model
local components. No new questionnaire/behavioral validity or whole-model score
is claimed. The two queries are neuroticism and confirmation-seeking, not Big5 C.

## Matched computation

The source teaching panel, Gaussian kernel, GRAIN input map, group normalization,
origin and regularizer are exactly those of NEXT82. The 28 observation rows
contain eight values, sixteen BODY derivatives and four context derivatives.
The source output basis uses all numerical rank of these vector targets.
Only the differential reader is learned; the common reader remains the same
source constant. This card does not close the common-reader or full-L interfaces.

Two coefficient solves in the same756-coordinate space:

1. Vector-only J0, a reparameterized NEXT82 GRAIN field control.
2. J1=J0+normalized scalar-projection loss+normalized K+V-projection loss.

Each author's five original source rulers (common evidence and four common
K+V directions) are used for all seven of that author's observations. They are
fixed functional probes, including at clamped teaching states, not newly
captured evidence there. No coefficient or basis used new8 native targets.

The control reproduces prior coefficients to relative2.42e-13 and application
outputs within7.99e-15. All six earlier reader comparators are also retained.
Kernel rank28/output rank27; source target reconstruction residual2.35e-15.

## Source results

Residual SSE / corresponding source target energy, separately normalized per
value/BODY/context group:

| Loss | Group | Vector control | Observable teaching |
|---|---|---:|---:|
| Vector | Value | 1.98919e-4 | 1.42803e-4 |
| Vector | BODY | 3.86642e-5 | 3.02414e-5 |
| Vector | Context | 2.45676e-6 | 1.93497e-6 |
| Scalar projection | Value | 5.83251e-4 | 2.77510e-7 |
| Scalar projection | BODY | 3.95564e-5 | 5.45633e-8 |
| Scalar projection | Context | 6.78905e-6 | 3.20030e-9 |
| Action projection | Value | 8.36401e-4 | 6.92711e-6 |
| Action projection | BODY | 4.44786e-5 | 1.74632e-6 |
| Action projection | Context | 2.21379e-6 | 1.31667e-7 |

No zero-energy groups were omitted. These large relative reductions concern
already small source training errors, not independent-author recovery rates.

## Recipient results

Same `response_coordinates__source_value`, query-specific N-minus-confirmation
local scalar/action composition throughout:

| Quantity | Constant | NEXT82 GRAIN / matched control | Observable teaching |
|---|---:|---:|---:|
| Reader vector error / native energy | 0.229089 | 0.226100 | 0.225873 |
| Assembled scalar SSE / zero | 0.193610 | 0.179712 | 0.178670 |
| Assembled scalar r | 0.577035 | 0.618477 | 0.612673 |
| Scalar author-centered MSE | 0.030494 | 0.028278 | 0.028411 |
| Anchored K+V SSE / zero | 7.288066 | 7.188125 | 7.202385 |
| Anchored K+V r | 0.677814 | 0.678197 | 0.678400 |
| Fixed-native-evidence differential scalar r | 0.474821 | 0.378854 | 0.342989 |
| Fixed-native-evidence differential scalar SSE / zero | 0.111139 | 0.117340 | 0.119390 |

Small raw scalar gain, slightly weaker ordering and slightly worse action error
are distinct outcomes. J1 does not supersede all older fields. All queries,
modes, four factors and four component compositions remain in the ten tables.

## Where error resides

Write t=d_native-d0 and S=row(source vector targets). For these restricted fields,

```text
e = e_in + e_out
e_in = P_S(dhat-d_native)
e_out = -(I-P_S)t.
```

The vector account is orthogonal: control .287700+.215128=.502828;
functional .287195+.215128=.502323. No reweighting within S can remove the
.215128 outside-vector floor. That does not make it a lower bound on each scalar.

Native-common-evidence scalar projected squared-error account:

| Reader | Inside square | Outside square | Signed cross | Total square |
|---|---:|---:|---:|---:|
| Vector control | 0.268764 | 0.303943 | -0.363950 | 0.208757 |
| Functional | 0.267052 | 0.303943 | -0.358589 | 0.212405 |

Inside error falls, but reduced cancellation makes total scalar error increase.
For the four K+V directions, cross-terms are positive instead. Both facts must
survive interpretation; orthogonality in vector space does not imply orthogonal
effects after a one-dimensional observation. These post-freeze accounts never
enter model construction or application.

## Development enabled

In pure vector regression with a scalar input kernel, restricting output to
row(Y) is sufficient. Once fixed directional rulers R enter the loss, that
restriction need not contain the unrestricted optimum. The sufficient source
space becomes span(Y,R) under the present isotropic output penalty. This is a
derivation about the objective, not proof the extension improves new8.

A post-account2D test gives restricted optimum(2/3,0) and full optimum(5/8,1/8),
with objective2/3 versus5/8. The missing direction is supplied by the evidence
ruler, despite the target lying entirely on the first axis. No empirical fit
was altered after this observation. See [theory](../../docs/GRAIN_OBSERVABLE_READER_TEACHING_THEORY_V1.md)
and [NEXT84](../../docs/GRAIN_SOURCE_OBSERVABLE_SPAN_PLAN_V1.md).

## Verification and artifacts

- 58 focused primary tests and one additional mathematical witness pass; Ruff
  and formatting pass. Not a full-suite result.
- Two source fits;10CSV summaries,8application and8evaluation packets reproduce
  exactly in cached replay. No unexpected application data reads.
-128anchored and128diagonal finite differences, max4.79e-10. All earlier six
  comparator outputs reproduce exactly; new control differs at most7.99e-15.
- Normal-equation relative residuals2.85e-16 and1.92e-15; independently tested
  against augmented least squares on synthetic observations.
- Canonical staged CPU including cached replay:7.820267seconds. Later preflight
  hashing, tests, publisher/audit and terminal exploration are outside that timer.
  No failed empirical stage or outcome-dependent runtime/code correction.
- Zero new neural/API/human-label/MEPS calls; no old scorer/lockbox edit or push.

[Audited publication and hashes](observable_reader_teaching_v1/publication.json).
Private artifacts: `results/grain_vector_mechanisms/logic_observable_reader_teaching_v1/`.
Runner: `scripts/run_grain_observable_reader_teaching.py`, stages prepare,
models, predict, evaluate, predict/evaluate with `--folder replay`, then audit.
The publisher exposes cohort tables only. Do not overwrite frozen stage folders.
