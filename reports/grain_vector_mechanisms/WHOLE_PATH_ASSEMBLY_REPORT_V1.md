# NEXT55: complete-function assembly and a relocated downstream reader

2026-09-10. Finite pass44 COMPLETE; whole-system goal ACTIVE. The previous turn
made progress by explaining NEXT53's local repair and fixing this whole-path test.

## Main result and decision

The frozen source map preserves part of the model's author-score ordering, but
strongly compresses score differences and loses the complete GRAIN response.
A subsequent cache-only crossing locates the loss primarily in the changed
downstream gradient at the mapped state, not the transported GRAIN direction.
This identifies a concrete composition problem rather than another local fit target.

All three candidates fail both preregistered-for-this-run utility rules. No
candidate replaces the original full score. NEXT53's useful attention22 key
component remains valid at its original scope; it was not the full function.
NEXT56 will split the actual source intercept from the remaining correction,
without refitting or selecting a new layer, gain or basis.

These are fixed-model logit/derivative fidelity results on reused source4/new8
author materials, not questionnaire correlations or a fresh PANDORA benchmark.
Here N means neuroticism, C means **confirmation_seeking**, not Conscientiousness.

## Frozen complete-function test

Primary implementation was committed as `7462cb56099766989dca368ae748f27d9c7d04b0`
before preparation and outputs. The [plan](../../docs/GRAIN_WHOLE_PATH_ASSEMBLY_PLAN_V1.md)
retained three pointwise functions: identity, normalization-matched value-only,
and value-plus-GRAIN-response joint. Source4 uses leave-one-author-out maps;
new8 uses the same all4 source map. No source refits or human labels.

    Lhat_q = S_22:end(Psi(X16))
    tauhat_qf = grad S_22:end(Psi(X16)) : D Psi(X16)[D_f]

The original prefix0–16 and actual suffix22–end remain native; only blocks17–21
are bypassed. No source Qbar/rbar surrogate is inserted in this suffix. Four
frozen whole:F01–F04 BODY directions use identical addresses/units for both queries.
The fixed logit L is distinct from the older probability score F and human Y.

One suffix reverse gives its full gradient and all four contractions. The pure
prediction process reads H16/layout and source maps, not current original pre22
states, gradients, scores or tau. Original NEXT51 score/tau caches open only after
all candidate outputs freeze. Source constants are evaluation baselines, not
offsets added to predictions. Four target-read denial probes pass; no extra denial.

## Complete scores

New8 authors, each query separately. Error ratio is SSE divided by the same-query
source-only constant's SSE. A usable score requires positive r and ratio<1 for both.

| Candidate | N r | C r | N error ratio | C error ratio |
|---|---:|---:|---:|---:|
| Identity shortcut | .019127 | .232814 | 34.018038 | 21.327860 |
| Matched value-only | .554093 | .422207 | 17.417794 | 7.668865 |
| Joint | .548248 | .419759 | 17.590959 | 7.785062 |

Joint N/C score SDs are .056261/.056683 versus native .903688/1.038604. Its
mean errors are -3.720160/-2.714723. Thus both origin and between-author amplitude
are wrong; a positive ordering correlation does not recover the original score.
Joint query-contrast r=.192888, mean=-.006711 versus native .998726, error
ratio=3.805550. No post-result centering/rescaling is used as a promoted predictor.

Source4 joint N/C r=.790348/.770249, but error ratios14.789364/4.093426 also
fail. The other source rows are retained in the tables, not mixed into new8.
Relative to the matched value-only control, joint score MSE worsens new8 by
.9942%/1.5152%; its previous local component benefit does not compose here.

## Complete GRAIN responses

New8, four directional actions per author. These are32 actions, not32 independent
people. A usable response requires positive r and SSE/zero<1 for both queries.

| Candidate | N r | C r | N SSE/zero | C SSE/zero | Paired query-contrast r | Contrast SSE/zero |
|---|---:|---:|---:|---:|---:|---:|
| Identity | .852449 | .547216 | 2.054720 | 7.676643 | .774237 | .403938 |
| Matched value-only | .061286 | .031736 | .998977 | 1.067422 | .060222 | .994885 |
| Joint | .049410 | .022457 | 1.003185 | 1.073457 | .057323 | .995280 |

The identity shortcut retains a useful *query-contrast* association, while its
per-query amplitude is excessive. Its four contrast-factor correlations are
.729685/.808905/.762517/.748285, but only17/32 action signs agree. Joint contrast
RMS=.022519 versus native .468223 (4.81%); its near-zero error ratio is not a
successful recovery of these differences. Full factor tables retain sign and scale.

Source4 query contrast instead has identity r=-.110941, joint=-.535686; respective
SSE/zero=1.351920/1.075029. This cohort difference stays visible. Joint's new8
per-query response error is .4213%/.5653% worse than the matched value-only control.

## Post-result crossing: state location versus direction transport

The [diagnostic note](../../docs/GRAIN_WHOLE_PATH_RELOCATION_NOTE_V1.md) and script
were committed as `0eefd04fffbd3dcb49b3b13aa3f992d01ec8527b` after primary results,
before this crossing. This is exploratory follow-up, not a retroactive hypothesis.
Use saved complete gradients g_I/g_J and frozen tangents d_I/d_J:

    II=g_I:d_I, IJ=g_I:d_J, JI=g_J:d_I, JJ=g_J:d_J.

| State / tangent | N action r | C action r | Query-contrast r | Contrast SSE/zero |
|---|---:|---:|---:|---:|
| I / I | .852449 | .547216 | .774237 | .403938 |
| I / J | .853034 | .547554 | .774968 | .403092 |
| J / I | .046924 | .021404 | .057138 | .995309 |
| J / J | .049410 | .022457 | .057323 | .995280 |

Holding the identity state and changing only its tangent preserves the response;
holding the identity tangent and moving the state loses it. The joint tangent
changes by only1.8196% in norm, retaining99.6639% of identity tangent norm.
In contrast, full-gradient norm ratios J/I are .196870/.209560 and cosines
.204444/.246966 (N/C). These are means of per-material ratios, not one pooled norm.

The symmetric state projection onto total action change is1.001422/1.001142;
tangent projections are slightly negative. For query contrast,1.002176/-.002176.
Values above1 or below0 express cancellation, not causal psychological shares.
Shrinking the identity response reduces per-query squared error, but destroys
the query-contrast correspondence; lower raw error alone would conceal this.

Within-material token-constant correction energy is99.9546% on new8,99.9534% on
source4. This does **not** yet mean the learned source intercept alone causes
relocation: the token mean of the other features can also contribute. NEXT56
distinguishes that exact source coefficient from the non-intercept remainder.
Crossed IJ/JI are diagnostics, not compatible stand-alone function derivatives.

## Reproducibility, resources and artifacts

- 72 candidate forward/reverse pairs,2 JVP checks,4 central-FD forwards;54.339960s
 native compute,8.177673s preparation/numeric work,4.966905s loading. Caps180/60s.
- 24 complete prediction packets;72 score rows,288 actions,36 score contrasts,
 144 action contrasts.16 primary tables replay exactly. Gradient contraction
 maximum difference2.9843e-13; independent metric error4.9960e-15. Both JVP/FD checks pass.
- Crossing384 scalar cells/192 contrasts,48 new off-diagonal vectors;1.534578s
 primary and1.362819s replay.24 arrays and14 tables exact; diagonal error0,
 independent metric/accounting errors3.9968e-15/2.6645e-15. Zero new neural calls.
- 37 focused tests pass with18 existing torch/Python3.14 deprecation warnings.
 Ruff and whitespace checks pass. No full-suite claim. The new crossing-auditor
 entry point initially omitted the experiment subfolder; corrected before a
 successful audit. No scientific output, fit or model run was repeated for this.

Private inputs/maps/gradients/author rows stay in ignored
`results/grain_vector_mechanisms/logic_whole_path_assembly_v1/`.
Public [primary tables](whole_path_assembly_tables/inventory.json) and
[crossing tables](whole_path_relocation_tables/inventory.json) contain12+9 cohort
aggregate CSVs and audit summaries only. No raw text, identity linkage or human labels.

Cache-only verification from the independent repository:

```bash
"../project persona/.venv/bin/python" scripts/audit_grain_whole_path_assembly.py
"../project persona/.venv/bin/python" scripts/audit_grain_whole_path_relocation.py
"../project persona/.venv/bin/python" scripts/report_grain_whole_path_assembly.py
```

Primary script `run_grain_whole_path_assembly.py` supports prepare/predict/evaluate;
do not rerun predict into the frozen packet. No current model job, new training,
API calls, scoring/release/lockbox/MEPS changes or remote push in this pass.

## Mechanism interpretation and next decision

Representation matching and functional substitution are distinct tests in the
model-stitching literature. Our full-suffix result follows that experimental
distinction rather than treating a local high correlation as end-to-end success.
See [Csiszarik et al., 2021](https://proceedings.neurips.cc/paper/2021/file/2cb274e6ce940f47beb8011d8ecb1462-Paper.pdf)
and [Bansal, Nakkiran & Barak, 2021](https://arxiv.org/abs/2106.07682).

The local interpretation is supported by the actual crossed gradients, not by
those papers proving a personality mechanism. The resulting
[theory](../../docs/GRAIN_WHOLE_PATH_ASSEMBLY_THEORY_V1.md) distinguishes score
origin, author-dependent scale and state-conditioned response. The
[NEXT56 plan](../../docs/GRAIN_SOURCE_OFFSET_MECHANISM_PLAN_V1.md) tests exactly
two additional source-defined functions in the complete suffix, then returns
to the dependency-reduction decision. No second fixed-basis optimization loop.
