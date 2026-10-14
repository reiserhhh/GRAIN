# F07-05, F09-05 and F13 recovery 2: channels, rivals, missingness, variance, MDC

Date: 2026-09-26 JST. Worker: Claude Opus 5.5. Three bounded packets that close
census gaps named after Fable seminar 01. No neural calls, API or private data.

## F07-05: recovery under measured reader emissions

`results/grain_gptpro_foundation_v3/F07/recovery_measured_r1/` (decision.md,
OPTIMIZER_CHECK.md). Information under measured emissions existed (F07 reader
link); recovery did not. 600 authors x 3 seeds x reader x generating channel,
MAP with the F07 prior, four estimators.

- Lexical reader (MCAR, perfect polarity): all channel-aware and naive estimators
  coincide exactly; the cost is information only (+.043 MSE_b vs oracle).
- NLI reader (MNAR: y=1 read as missing 75% in source): modelling the measured
  channel beats naive ignorable reading by -.029 [-.034,-.025] (application
  channel) and -.019 [-.025,-.014] (source channel); naive reading is biased
  (b -.133, coverage .852). The measured source-to-application channel shift
  (.1875) costs +.011 [.007,.015] MSE_b and lowers coverage .939 -> .910.
- Rule: fit the full emission law, including response-dependent missingness;
  re-estimate it on independent pilot material when reader or material changes.

## F09-05: recovery tables and exact extra-observation control

`results/grain_gptpro_foundation_v3/F09/recovery_tables_r3/` (r1 tie-break
artefact and invalid r2 retained). From frozen author packets only.

- Family recovery .791 [.758,.820] (order-invariant tie credit), beta .529
  [.482,.576], joint .402; 22.2% of authors have exactly tied family posteriors
  (development data uninformative between families); high-confidence wrong
  family .036; in-family false rejection .038; M4 detection .027 [.007,.053]
  (no demonstrated power at 12 events: INCONCLUSIVE, not absence).
- Exact positive control: process-class Bayes accuracy .500 from corner reports
  at every budget, .502-.503 with intermediate levels, .642/.690/.727 with the
  belief observed (1/2/3 reps).

## F13 recovery 2: dynamic missingness, variance components, MDC95

Protocol `docs/GRAIN_F13_RECOVERY2_PROTOCOL_V1.md`; run
`results/grain_gptpro_foundation_v3/F13/recovery2_r1/`. Scorer
`f13_dynamic_missing.py` (exact joint-Gaussian conditioning on the observed
subset; validated against archived complete-data outputs to 6e-16).

Dynamic targets (F05 core, 200 authors x 3 seeds; estimator assumes ignorable
missingness; MSE differences vs complete, paired author bootstrap):

| mechanism (mean observed of 48) | T2 state | T3 planned future |
|---|---|---|
| MCAR .2 (38.4) | +.031 [.014,.049] | +.006 [-.000,.012] INCONCLUSIVE |
| condition .2/.6 (28.9) | +.111 [.081,.142] | +.023 [.010,.036] |
| MNAR .2+.4 sigmoid(y) (28.8) | +.071 [.045,.097] | +.015 [.003,.029] |

MNAR biases the level-bearing forecast (T3 bias -.064) but not the state (T2
-.009); intervals widen so Wilson gates pass (coverage .93-.95), whereas the
complete-data T2 reference keeps its known failed gate in one seed. Rule:
missingness mechanism effects are target-specific; an unbiased state score can
coexist with a biased forecast under the same MNAR process.

variance_components.csv (covariances kept; truth from the declared simulators):
F04 L recovers every facet within its source-bootstrap interval (e.g. person
.988 [.838,1.139] vs 1; slope .234 vs .25; residual .253 vs .25). F05 core
recovers the stationary state variance (.377 vs .373) but mis-splits it: state
innovation .301 [.284,.321] vs .25 and residual .183 [.166,.197] vs .25 (seed
20260915; same direction all seeds). The state/noise split is weakly identified
even when their combination is not. Text reader and material facets are
NOT_IDENTIFIED separately (paired source/channel draws; no material resampling).

MDC95 for T1 from two 16-event occasions of the same author (true change 0):
noise-predictive MDC95 .84 raw (.82-.89 reference SD); fraction within MDC
.955/.895/.950 (seed 20260916 just outside its Wilson interval, .845-.930). The
posterior-variance formula sqrt(V1+V2-2Cov) is about 5% larger and conservative
(.970/.920/.965). An observed T1 change must exceed about .85 reference SD before
it exceeds measurement error at this design; MDC is not a clinical MCID.

## Requirement evidence

R06/R13 (F07-05 recovery under measured MNAR channel), R18/R19 (F09 separated
recovery, power boundary for M4), R24/R29/R12 (dynamic missingness, variance
components, MDC on actual runs), A06 (variance facets). Remaining F13 gaps:
binary next-p missingness, B2/M parameter uncertainty, null+partial-bounds
deviation decision. No parent card promoted.

## Addendum (2026-09-26, later): F09 M4 exact power bound and targeted contrast

- **`F09/m4_power_bound_r1`:** at the 24-event design, the exact
  Neyman-Pearson upper bound on the power of any level-.05 test against M4 is
  .207/.450/.716 (prior-mixture null) or .297/.564/.830 (every-atom level),
  for beta .5/1/1.5. The budget was therefore not the limit; the omnibus
  statistic (.027) was.
- **`F09/targeted_m4_r1`** (protocol `docs/GRAIN_F09_TARGETED_M4_PROTOCOL_V1.md`,
  frozen before scoring): a targeted likelihood-ratio contrast with exact
  composite calibration.
  - Level: <= .043 at every in-family atom.
  - Exact power: .119/.374/.658.
  - Stored application M4 authors: detection .36 [.29,.44], against .027 for
    the omnibus check.
  - In-family false rejection: .040/.047.
- **Rule:** target out-of-family checks at declared rivals. Non-rejection never
  certifies membership.
