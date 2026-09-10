# NEXT56: the source intercept relocates the complete downstream reader

2026-09-10. Pass45 COMPLETE; whole-system goal ACTIVE. Primary freeze
`54892cc297f70ddb103d39d656853787354af935`, before preparation and model outputs.
The previous goal turn made progress: complete-function assembly plus reader crossing.

## Finding and decision

The actual source intercept largely reproduces NEXT55's shifted downstream
reading: offset-only gradients have new8 cosine .9998 with the full joint map.
Keeping only the non-intercept correction preserves the identity-like reader
and increases query-contrast response r from .774 to .811, with13.18% lower
contrast squared error. This is a retained complete-response contrast effect.

All four functions still fail the frozen per-query whole-score/whole-response
utility rules. Source4 contrast stays negative. The hidden-state substitution
family therefore closes here; no bias/gain/rank search follows. NEXT57 changes
the approximation object to one scalar full-output function and its derivatives,
aiming to remove the actual suffix, rather than imitate every intermediate state.

Targets are fixed-model logit L and GRAIN actions tau, not human questionnaire Y.
N=neuroticism; C=confirmation_seeking, not Conscientiousness. The12 reused author
materials are a source4/new8 mechanism panel, not a new PANDORA benchmark.

## Fixed experiment

The pointwise source features [p,1,s,s^2] have an explicit unit feature, center0,
scale1, whose coefficient is b. With V(X)=Z(X)B-1b^T:

| Function | State into actual suffix22–end | Compatible direction |
|---|---|---|
| I | X16 | D |
| O: offset-only | X16+1b^T | D |
| V: non-intercept-only | X16+V(X16) | D+DZ[D]B |
| F: previous joint | X16+1b^T+V(X16) | D+DZ[D]B |

Source4 uses leave-own-author-out maps; new8 uses the same all4 source map.
I/F reuse NEXT55 outputs without fresh model calls. Only O/V are new. States and
derivatives obey F=O+V-I; nonlinear suffix outputs need not. H16/BODY addresses,
four whole:F01–F04 directions, observer/FP32/MPS, queries and c stay unchanged.
Current original targets and old candidate outputs open only after prediction
freeze. No new fits, labels, gain calibration or current-target add-back.

## Complete scores and responses

New8; score ratio=SSE/source-constant SSE, response ratio=SSE/zero. Each frozen
utility rule requires both query r>0 and both corresponding ratios<1.

| Function | N score r | C score r | N score ratio | C score ratio | N response r | C response r | N response ratio | C response ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| I | .019127 | .232814 |34.018038|21.327860|.852449|.547216|2.054720|7.676643|
| O | .542815 | .419178 |17.361165|7.622517|.067433|.041886|.996836|1.061417|
| V | .004455 | .242301 |35.772985|22.242016|.854279|.543932|1.885496|6.801037|
| F | .548248 | .419759 |17.590959|7.785062|.049410|.022457|1.003185|1.073457|

O reproduces F's compressed score SD: .054261/.056105 vs F .056261/.056683,
native .903688/1.038604. Its mean errors remain -3.694121/-2.682123. V instead
retains excessive identity-like action amplitude and worsens whole-score bias.

Raw query contrasts are paired before metrics.32 directional cells come from8
authors, not32 independent people. Native new8 contrast RMS=.468223.

| Function | New8 contrast r | SSE/zero | RMS | Same signs | Source4 contrast r | Source4 SSE/zero |
|---|---:|---:|---:|---:|---:|---:|
| I |.774237|.403938|.382640|17/32|-.110941|1.351920|
| O |.050688|.995830|.022886|15/32|-.522423|1.073732|
| V |.810727|.350690|.404424|17/32|-.088634|1.325876|
| F |.057323|.995280|.022519|16/32|-.535686|1.075029|

V's four factor contrast r values=.799043/.843184/.795780/.782092; SSE/zero=
.321066/.333970/.330799/.402831. All improve over I's contrast error. Signs do
not improve; source4 aggregate remains negative. All source/query/factor rows stay
in the tables, including source4 score failures and weak/negative factor cells.

## Ingredient mechanism

New8, means of per-material geometric ratios:

| Quantity | N | C |
|---|---:|---:|
| O gradient cosine with F |.999819|.999807|
| O gradient relative difference to F |.021124|.022102|
| O gradient norm / I |.195040|.207390|
| V gradient cosine with I |.999264|.999319|
| V gradient relative difference to I |.045676|.043806|
| V gradient norm / I |1.024049|1.022796|

O's input tangent is exactly unchanged. V has exactly F's tangent, changing
1.8196% from I. The explicit source intercept reproduces reader relocation with
a compatible function, extending NEXT55's crossed-gradient diagnostic.

Offset energy/total correction energy=.999093/.999108; remainder=.000460/.000461;
signed cross=.000447/.000431. Terms sum to1, but are not information fractions.
b differs from the material-mean correction by .2542%/.2543% in norm, cosine
.999997. This identifies the coefficient behind the earlier token-constant statistic.

Symmetric factorial response-change projections: offset .985092/.972044,
remainder .014908/.027956. Query contrast instead has1.027882/-.027882: the
remainder opposes offset suppression. Interaction RMS=.038168 vs total contrast
change RMS=.374840. Score interaction means are .110310/.072357. These exact
signed accounts are neither additive-suffix assumptions nor psychological shares.

## Verification and artifacts

- 48 new forward/reverse pairs,2 JVP checks,4 central-FD forwards;39.669475s
  native,5.735351s preparation/numeric,3.440526s loading.120/60s caps respected.
- 24 new packets;96 scores,384 actions,48 score contrasts,192 action contrasts.
  Four target-denial probes,0 extra denials; prior candidate outputs excluded
  from prediction. Original targets/source-constant joins independently checked.
- 26 CSVs replay exactly. Contraction error<=2.9843e-13; state identity<=1.4211e-14;
  metric/factorial errors<=3.5528e-15; energy identity<=4.548e-16. JVP/FD checks pass.
- 41 focused tests, including4 new tests, pass;18 existing torch/Python3.14 warnings.
  Ruff and whitespace checks pass. Pre-freeze lint findings (shebang/import spacing)
  were corrected before execution. No scientific restart or target-fit rescue.

Private: `results/grain_vector_mechanisms/logic_source_offset_mechanism_v1/`.
Public: [14 aggregate CSVs and audit](source_offset_tables/inventory.json).
Core: `grain_research/vector_mechanisms/source_offset.py`; runner and verifier:
`scripts/run_grain_source_offset_mechanism.py`, `scripts/audit_grain_source_offset_mechanism.py`.

```bash
"../project persona/.venv/bin/python" scripts/audit_grain_source_offset_mechanism.py
```

This is cache-only. Do not rerun predict into the frozen packet. No current model
job, API/human Y, push, scorer/release/lockbox/MEPS change or private text export.

## Mainline consequence

RMSNorm does not generally remove translations, a relevant mathematical route
but not proof that it alone caused this full-suffix result.
[Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467).
Representation matching and functional substitution are distinct tasks:
[Csiszarik et al., 2021](https://arxiv.org/abs/2110.14633).
NEXT57 directly approximates output plus derivative, following the principle of
[Sobolev training](https://arxiv.org/abs/1706.04859), not another hidden-state grid.
Read [theory](../../docs/GRAIN_SOURCE_OFFSET_MECHANISM_THEORY_V1.md) and
[NEXT57](../../docs/GRAIN_COMPLETE_FUNCTION_EXTRACTION_PLAN_V1.md).
