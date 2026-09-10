# NEXT76: construct the content-value channel from source H16

2026-09-11. PLANNED after NEXT75, not yet run. One dependency-removal card, not
another selector/rank/alpha sweep or a new personality accuracy benchmark.

## Question and decision

The source-trained selection function is usable; its directional calibration is
mixed. Current V remains a native side input. Can a source-only function supply
that content channel, and does differentiating it reveal a useful second route
beside the key-selection covariance? This is a different missing dependency,
not an attempt to declare NEXT75 a universal winning model.

This does not reopen NEXT55-56 full hidden-state substitution. Predict only the
projected attention value channel, use the frozen numerical selector, and run no
native suffix. An offset in V has a different role from moving a hidden state
through all downstream nonlinear blocks; report its actual scalar/action effects.

## Fixed construction

1. Keep source4LOO/new8all4, both queries, all4 `whole:F01-F04`, both own/pooled
   readouts. Reuse each split's NEXT75 rank32 basis and source feature scales;
   no additional basis/rank search. Retain both the identity selector and fixed
   NEXT75 selector; neither is selected using the new8 result.
2. Source teacher V/dV come from existing attention22 state/direction captures
   and fixed RMSNorm/V matrices. No new neural capture. Derive H16 identity
   V0/dV0 using the same matrices. Do not import current native K/Q/V into
   application. Separate g-only packets preserve the currently retained readout;
   record their exact lineage rather than claiming that dependency removed.
3. Fit Vhat(H)=V0(H)+Phi_S(H) C_Sq and its exact derivative to source V/dV.
   Phi is the ordinary67-column causal-context feature function, not the
   per-sequence logit gauge projection. Its value constant remains meaningful.
   Equal authors and allowed tokens, four equal directions, source residual-RMS
   normalization, alpha=.01; one joint solve per reference/query, ten total.
   No new8 tuning or rescaling. Preserve the source intercept contribution in
   reporting; do not subtract it after observing an undesirable effect.
   Implementation pin: the explicit constant feature is unpenalized; all other
   coefficients have alpha=.01. Report its bias-column contribution, not a claim
   that it equals every constant induced by the source input/feature centers.
4. Two value candidates: identity V0 and learned Vhat. Cross with the two fixed
   selectors. Prior native-V selector outputs are immutable comparators. Main
   application reads H16/layout/fixed weights/source models/g-only, no current
   targets, native V/dV/K/dK/Q or previous predictions. Freeze outputs first.

## Common function and outputs

    S(H)=sum_h E_(a_h(H))[v_h(H)]
    D S[D_f]=sum_h Cov_(a_h)(v_h,D ell_h[D_f])
             +sum_h E_(a_h)[D v_h[D_f]].

Export scalar, key-selection partial, content-value partial and total for every
condition. Native/source-Q key-only actions are not interchangeable with the
new total target. Evaluation reconstructs matching native V/dV totals from old
captures; it also retains both query choices and each component separately.
Both query targets hold Q fixed when differentiating K and V; the native DQ
channel remains outside these totals. Native-Q K/V partials must reproduce the
already frozen NEXT49 channel outputs before comparisons are published.
The retained fixed g is constant for this derivative, not a full model suffix.

Metrics: source V/dV residual losses; equal-author value/dvalue errors; all-query,
factor, contrast and mode r, RMS, signed agreement, SSE/zero for key/value/total;
scalar fidelity to the same local functional (not final L or human Big5 scores).
Record signed cancellation, rather than implying positive component energies add.

## Execution and finish

- Suggested module `grain_research/vector_mechanisms/source_value_channel.py`,
  runner `scripts/run_grain_source_value_channel.py`, matching tests/publisher.
- `results/grain_vector_mechanisms/logic_source_value_channel_v1/`; public
  aggregates/report plus dependency/theory updates, no author/text/token exports.
- Freeze before new8 application; source exclusion, g-only packet tests, exact
  scalar finite differences including Dv, component-sum/replay checks.
- Ten fits, zero neural/API/Y/MEPS;180CPU seconds including one numeric replay.
  Stop after the fixed card; no capacity rescue, new factor search or old queue.
- If the content function works locally, retain that dependency reduction and
  plan a source-fixed readout compilation. If it fails, compare source fit and
  source-disjoint action errors to identify missing support/coupling, then record
  one new mechanism question rather than restarting a value reconstruction grid.

The overall goal stays ACTIVE. Clinical/TA interpretation remains a downstream
source-backed interface; no automatic psychoanalysis, new clinical data use or
external-label fitting is included in this numerical card.
