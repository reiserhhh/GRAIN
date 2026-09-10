# NEXT81: fix the last context and test material-conditioned reader motion

2026-09-11. PLANNED, not executed. NEXT80 adds real source-reader directions but
does not restore opposing response modulation. This card distinguishes omitted
non-last material dependence from a context-only function. No new predictor fit,
bandwidth grid, coefficient search or new8 target-based choice.

## 1. Decision changed by this experiment

Can the personality-query differential reader be represented by the paired last
context alone, or do the actual material and GRAIN changes also alter the reader?
This is directly the same-material interface: evidence content can change both
what is read and how it is read. If a material channel exists, develop it before
expanding context-only interpolation again.

## 2. Fixed source-only intervention

Use the same four source authors and N/C query rendering, H16 states, GRAIN BODY
directions, native scalar L, block22 post-attention gradient hook and observer
weights as NEXT80. C denotes confirmation-seeking, not conscientiousness.

First reproduce two unperturbed parity readers (first source, both queries)
against NEXT80. Then define one all-four source reference, separately per query:

```text
lastbar_q = mean_i H16_iq,last
Hstar_iq = H16_iq with its complete last row replaced by lastbar_q
rstar_iq = grad_(post22,last) L_q(suffix17..end(Hstar_iq))
dstar_i = rstar_iN-rstar_iC
```

Both query-specific full last vectors are now constant across the source panel,
not merely their mean. All other rows, BODY, length, layout, mask and positions
remain those of the original source. Residual differences therefore concern
**non-last material plus retained layout**, not pure personality or pure BODY.
The clamp is a hidden-state intervention, not a claim of naturally occurring text.
Record source displacement norms; do not tune the clamp after viewing outcomes.

Eight clamped reader calls, no LOO learning or recipient application this card.
Compare original and clamped d dispersion, cosines, scalar/action projections and
query-specific reader variation. For any c-only deterministic reader on this
clamped panel, the best constant differential-vector SSE is exactly:

```text
min_d0 sum_i ||dstar_i-d0||^2 = sum_i ||dstar_i-mean_j dstar_j||^2.
```

This finite-panel approximation floor can locate missing inputs. Do not call its
ratio an author/population causal variance share; native layout and non-last
query representations are still retained. The source unit remains four authors.

## 3. Direct GRAIN-to-reading intervention

Within each clamped state, perturb each of the four **already frozen** BODY
directions separately by +/-eta D_if, eta=.01. D is the old relative-RMS-scaled
`body_directions` output, retained from the original state; last and other rows
remain zero in D. No factor selection, new names or amplitude optimization.

```text
jBODY_if = [(rstar_iN(+eta D_if)-rstar_iC(+eta D_if))
             -(rstar_iN(-eta D_if)-rstar_iC(-eta D_if))]/(2 eta)
```

Keep query-specific directions in their original query coordinates if needed;
their correspondence is the same-material factor/address contract, not silently
assuming identical hidden arrays. This takes64 native calls. For each coded
factor report vector slope RMS, plus/minus change size, consistency of its sign
after projection onto fixed native and all four prior numerical evidence vectors,
and its overlap with the old context-slope range. Preserve all four factors.

The slope is the derivative of a post22 endpoint gradient with respect to an
upstream BODY perturbation. It is not a human causal effect or automatically an
extra term in original L's first derivative. A later numerical scalar depending
on BODY must declare its target and use the full product rule; no double-counting
reader curvature as an additive native attention attribution.

## 4. Budget, checks and branch

- At most78 native reverse calls:2 parity +8 fixed-last +64 BODY perturbations
  +4 half-step checks (first source, first factor, both queries, both signs).
- One frozen local observer load,120 native-compute seconds and90 numeric CPU
  seconds. Track native/loading/failed/successful costs separately. Stop and
  preserve the complete unit at cap, do not silently restart a long queue.
- Parity relativeL2<=1e-3 and maxabsolute<=1e-4, as NEXT80. Record the one
  half-step discrepancy; if >.20, label that secant unreliable and close the
  derivative estimate rather than searching eta on the outcome.
- Verify full-last equality by query and zero BODY perturbation at the last
  row. Record code/model/source hashes before native work. One cached numerical
  replay; aggregate-only outputs, no current new8 captures or labels.
- If clamped differences and GRAIN-reader motion are both nonzero beyond parity
  and the measured numerical discrepancy, develop d(B,c,layout) with source
  material derivative teaching. Report magnitude, not binary significance on4.
- If only clamped differences remain, preserve a non-last/layout input gap but
  do not claim the four GRAIN directions exhaust it; use their residual geometry
  to choose one next input. If both are negligible, test finite context curvature
  over one fixed source path. No epsilon, rank, kernel or clinical-label sweep.

## 5. Deliverables and scope

Suggested module `reader_material_dependence.py`, runner
`scripts/run_grain_reader_material_dependence.py`, focused tests; private
`results/grain_vector_mechanisms/logic_reader_material_dependence_v1/`;
aggregate `READER_MATERIAL_DEPENDENCE_REPORT_V1.md` and theory update. At minimum:
clamped reader dispersion, c-only constant floor, factor-coded BODY-reader slopes,
projections/range overlaps, parity/step checks, provenance and exact cost.

No new human/MEPS data, API, Claude, model download, publication of private
vectors/texts, old lockbox/scorer modification or new8 prediction fitting. H16
remains a neural-derived representation. This is a finite next mechanism card,
not universal logic exhaustion. Goal ACTIVE.
