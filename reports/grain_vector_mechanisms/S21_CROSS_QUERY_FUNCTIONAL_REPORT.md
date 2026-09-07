# S21: shared authors do not imply a universal query reader

2026-09-08. COMPLETE:12 reference and12 transfer authors, four digit0-9 queries,
216 new forwards/108 backwards,411.41seconds after loading. N references/effects
were reused from S19/S20; no new human labels.

## Main finding

Own-query functional directions transfer to other authors. Cross-query substitution
usually loses substantial effect fidelity despite the same bodies and answer
alphabet. N and confirmation retain a shared relationship, but are not identical.

Effect reconstruction r on the12 transfer authors; rows are recipient questions,
columns are donor reference gradients:

| Recipient | N | Confirmation | Feeling numeric | Content control |
|---|---:|---:|---:|---:|
| N | .9307 | .8266 | -.2561 | -.0716 |
| Confirmation | .7952 | .9486 | .1401 | .4750 |
| Feeling numeric | -.3672 | -.1471 | .8015 | .2074 |
| Content control | .1625 | .2688 | .2741 | .6374 |

Own-query bootstrap95%CI: N[.8811,.9890], confirmation[.7793,.9919],
Feeling[.3589,.9704], control[.1282,.9494]. Own-query MSE/effect-energy:
.3113/.0927/.3327/.5484. These are internal intervention effects, not personality
accuracy. Numeric Feeling is a separate condition, not the native T/F benchmark.

The N versus content-control substitution delta r=-1.0023,
pairedCI[-1.5833,-.3524]. The N versus confirmation delta=-.1042[-.2514,.0091];
N/confirmation sharing is useful, with different gain/bias (N using confirmation
has error ratio1.6367 despite r=.8266). Keep absolute error alongside correlation.

## Where the difference appears

Mean reference-gradient cosines: N/confirmation.8497, N/Feeling.3160,
N/control.3714. Recipient pairing vectors remain close: mean per-author
N/confirmation cosine.9825 and N/Feeling.9793. Actual own-query effects correlate
N/confirmation.7510 and N/Feeling-.1341 on transfer12.

This locates query dependence in the operative suffix direction as well as
in query-conditioned evidence; a universally shared numeric-answer covector is
not an adequate account of these four conditions. Identical output alphabets
alone do not explain the observed own-query advantage. Bodies/prefix tokens are
identical; question lengths were not equalized. This is a query-condition
comparison, not an isolated causal test of one semantic word.

Within-query first gradient energy fractions: N.9642, confirmation.9274,
Feeling.8790, control.8498. Source-specific variation remains. The smaller
gradient-reference size and12-author transfer panel differ from S20's24/24,
so do not present .9307 versus .9431 as a regression in the N mechanism.

## Formula and numerical evidence

R_u,q<-r=g_ref,r^T d_uq, with reference covectors estimated only from reference
authors. No transfer gradient, effect fit or questionnaire is used by R.
Path integrals use the internal completeness formulation in
[Sundararajan et al.2017](https://proceedings.mlr.press/v70/sundararajan17a.html);
sensitivity spectra use finite gradient-second-moment algebra motivated by
[Constantine & Gleich2015](https://arxiv.org/abs/1408.0545).
These are computational response functions, not newly named human factors.

Max reference completeness gap1.75e-6; native endpoint replay0; no adaptive
five-node unit was required. All score/membership/prompt hashes are frozen.
Six new tests include native-template replay, preserved raw bodies, old S1B
control-wording equality, cross-tensor orientation and no transfer influence.

Code/protocol3a42314. Private output:
results/grain_vector_mechanisms/cross_query_functional_s21_r1/.
Includes all96 author/query packets, readers_frozen.json, private_readers.npz,
pilot_004.json, cross_reader_metrics.csv, private_transfer_predictions.csv,
summary.json and run_status.json. No protected labels, released-scorer changes,
old queues, remote push or cloud text upload.

Reproduce with parent venv Python and a new output directory:
scripts/run_grain_cross_query_functional.py prepare
then run --max-reference-authors 4, inspect, run --max-reference-authors 12,
and evaluate. Full61 focused tests pass. Next: S22 measures these functions
on the actual reference distribution instead of relying on coordinate angles.
