# Evidence mass and relative selection: NEXT70

2026-09-11. COMPLETE; primary implementation frozen at `125e40e`.
Finite logic pass59; whole-system goal ACTIVE. Previous pooled-quadratic family
stays closed. This contrast does not replace the production scorer.

## Finding and decision

Relative selection learns a better **source neuroticism cross-token response**,
but that advantage does not transfer to the confirmation query or whole scores.
The mass-retaining unnormalized reader instead improves exposed new8 GRAIN
direction errors over matched local by approximately3.60%/5.41% (N/confirmation).
Keep that local functional improvement. Do not infer a universally superior pool
or choose a new per-query hybrid using these exposed eight authors.

The next mechanism question is which part supplies the change: direct content,
relative evidence competition, or log mass. NEXT71 is a bounded decomposition of
these already-frozen functions, not another training run or selector search.

## Exact comparison

Same source4 authors, original/last-shift/front-plus/front-minus views, two native
queries (`neuroticism`, `confirmation_seeking`). The latter is not conscientiousness.
Same source U8,64Gaussian landmarks/bandwidth; select indices0,4,...,60 for16kernels.
Keep original1169local features and add144coefficients per arm, total1313.

For kernel k, Z=mean(w), m=mean(w*z), r=m/Z, ell=logZ. Compare linear readout of
(m,ell) with (r,ell), source-original centering only. Same source score/density/
last-response/B-response objectives, each coefficient1, Ridge alpha=.01.
These are invertible coordinates for Z>0, but not the same penalized linear
function class. The comparison includes coordinate scaling and inductive bias;
it is not pure evidence that normalization adds information.

## Source cross-token response

Change only front BODY; measure gradient half-change on untouched back BODY B.
SSE/zero uses a predictor of zero **change**, not a chance personality baseline.
Lower is better; full-field column is the mean of author-level relative errors.

| Query | Reader | Coordinate SSE/zero | Full hidden field SSE/zero | Four GRAIN actions SSE/zero | Action r |
|---|---|---:|---:|---:|---:|
| N | matched local | 1.000000 | 1.000000 | 1.000000 | undefined |
| N | prior linear quadratic | .973321 | .995399 | .960824 | .226789 |
| N | prior nonlinear quadratic | .993427 | .998914 | .962056 | .214768 |
| N | unnormalized + mass | .995625 | .999275 | .969572 | .191214 |
| N | normalized + mass | .979114 | .996394 | **.948326** | .250701 |
| Confirmation | matched local | 1.000000 | 1.000000 | 1.000000 | undefined |
| Confirmation | prior linear quadratic | .974318 | .995102 | .961599 | .195567 |
| Confirmation | prior nonlinear quadratic | .995628 | .999207 | 1.058208 | -.242055 |
| Confirmation | unnormalized + mass | .996824 | .999466 | 1.045596 | -.240165 |
| Confirmation | normalized + mass | 1.001494 | 1.000216 | 1.042535 | -.143870 |

For N, normalization reduces matched unnormalized coordinate error1.66% and
GRAIN-action error2.19%. The N action error is5.17% below zero-change prediction;
predicted action RMS .00004153 vs actual .00025632, so amplitude remains small.
Confirmation has no corresponding positive source recovery. Its normalization
increases coordinate error while slightly reducing action error: the GRAIN
directions and full field weight different parts of the response.

## Exposed source-disjoint new8, original material

No new8 targets were used in fitting/application, but these eight authors have
been examined in earlier development. These are native model outputs, not human
questionnaire accuracy, not a new independent replication and not significance tests.

| Query | Reader | Whole-score r | Score RMSE | GRAIN action SSE/zero | Allocation SSE/zero |
|---|---|---:|---:|---:|---:|
| N | frozen original | .503312 | .821332 | 1.080006 | .932929 |
| N | matched local | .143813 | .919341 | 1.085740 | .932307 |
| N | unnormalized + mass | .151881 | .917737 | **1.046627** | **.923813** |
| N | normalized + mass | .142292 | .920007 | 1.096321 | .942940 |
| Confirmation | frozen original | .538314 | .903460 | 1.092412 | 1.027697 |
| Confirmation | matched local | .356246 | .975583 | 1.053263 | 1.026691 |
| Confirmation | unnormalized + mass | .349486 | .979483 | **.996247** | 1.006257 |
| Confirmation | normalized + mass | .356000 | .977113 | 1.019333 | **1.004318** |

Both mass arms reduce confirmation action error versus matched local; unnormalized
is stronger on total action, normalized slightly stronger on centered allocation.
Whole-score readout does not improve over the frozen original. A direction of
model sensitivity and the author's complete score are distinct targets.

## Reproducibility and files

- Four fits;0new neural/API/human-label/MEPS calls. Source-only input allowlists;
  current native scores/gradients denied during application.
- 68 focused tests/Ruff passed (two inherited SWIG warnings).384 actual GRAIN
  finite differences, max error1.9578072496528875e-10; direct source loss replay0.
- 48two-arm prediction packets and11CSV byte-identical on replay. All eight prior
  control candidates retained without refitting; staged CPU32.817521seconds.
- Six aggregate tables and provenance:
  [normalized_evidence_v1](normalized_evidence_v1/publication.json).
- Private artifacts: `results/grain_vector_mechanisms/logic_normalized_evidence_v1/`;
  models, native targets, inputs and author rows remain ignored/private.
- Implementation: `grain_research/vector_mechanisms/normalized_evidence.py`;
  runner: `scripts/run_grain_normalized_evidence.py`.

Stages, in a fresh output directory: `prepare`, `fit`, `predict`, `evaluate`,
`predict --folder replay`, `evaluate --folder replay`, `audit`; use `--stage` for
each. Existing frozen artifacts must not be overwritten. Publisher:
`scripts/report_grain_normalized_evidence.py`.

See [theory](../../docs/GRAIN_NORMALIZED_EVIDENCE_THEORY_V1.md) and
[NEXT71](../../docs/GRAIN_EVIDENCE_COMPONENT_PLAN_V1.md). H16 remains an encoder
dependency. Same-material auxiliary interpretation and human review remain a
separate evidence layer; no new clinical claim or old lockbox change.
