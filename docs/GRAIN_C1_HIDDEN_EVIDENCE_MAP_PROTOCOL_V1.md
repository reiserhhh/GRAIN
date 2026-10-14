# GRAIN-C1-01 hidden evidence map protocol V1

Frozen 2026-10-02 before any evaluation-seed output. Authority:
`docs/GRAIN_CLAUDE_RESEARCH_LOOP_HANDOFF_20261002.md` (GRAIN-C1-01) and the seminar plan
`reports/grain_fable_theory_review_20261002/THEORY_REVIEW_AND_NEXT_DIRECTION.md` (project persona).
Code: `grain_research/construction/hidden_evidence_map.py`, runner
`scripts/run_grain_hidden_evidence_map.py`, tests `tests/test_grain_hidden_evidence_map.py`.
Output root: `results/grain_construction_v1/C1/hidden_map_r1/` (new, non-overwriting).

## Question

Can a source-learned or pilot-anchored evidence map recover the target response, rather than a
stable but target-irrelevant author fingerprint, when the map from latent expression to observed
vectors is unknown? Simulation only. No text, no neural call, no human label.

## World (reuses F04 world L parameters)

Per author u: level a_u ~ N(0,1); slope b_u ~ N(.5,.5) (W1: b_u = 0); stable fingerprint
f_u ~ N(0, 2^2 I_4); style-by-context response r_u ~ N(0,1). Events t = 0..31 cycle the F04
conditions z in {-1,-1/3,1/3,1}; topic from the source family {0..7}. Expression
e = a_u + b_u z + N(0,.5^2) (F04 sigma .5). Observation in D = 64 dimensions:

x = 1.5 u_e e + U_f f_u + u_s r_u z + U_t[:,topic] + u_c z + N(0, I_64).

u_e is unit with half its energy on a public 8-coordinate lexicon (declared signs); U_f and U_t
are Gaussian / sqrt(64) (U_t times 1.5); u_s, u_c unit. Columns overlap (unknown mixing).
Dimension choice: D = 64 exceeds the latent count (1 + 4 + 1 + 12 + 1 = 19) so nuisance and target
overlap by chance, keeps pilot regression underdetermined enough to need regularization
(192 pilot texts, 144 train), and runs on CPU in seconds. Fingerprint variance per direction (4)
exceeds the target level variance on u_e (2.25), and style-slope variance (1) exceeds the target
slope variance on u_e (.5625): a pure stability criterion is designed to prefer the fingerprint.
Linear mixing only; nonlinear maps are deferred.

Worlds (same maps per seed; common random numbers across W2-W4):
- **W1_shared_level**: b_u = 0; target = level a_u. Pilot declares expression labels.
- **W2_conditional**: target = Delta_u = m_u(+1) - m_u(-1) = 2 b_u (primary); level secondary.
- **W3_missing_anchor**: as W2, but the pilot declares topic labels only (no expression anchor).
  Control: exposes the boundary; never used to pass W1/W2.
- **W4_confounded_anchor**: as W2, but pilot labels are confounded with pilot-writer fingerprints
  (writer component .9). Control for an anchor design error.

Samples: 300 source and 200 application authors x 32 events (F04 budget); pilot 12 writers x 16
texts, writers 0-8 train, 9-11 validation. Seeds 20261002, 20261003, 20261004.

## Information boundary

Estimator view (only): source x, z, opaque author index; application x, z, author index; pilot x,
writer index, declared label (or none) and label kind; public lexicon indices/signs.
Evaluator-only: maps, a, b, f, r, topics, true expressions, pilot latents. Hidden truth is not a
feature, model selector or tolerance tuner. Oracle arms are explicitly privileged.

## Arms (information and capacity differ; this is not a fair contest)

| arm | direction from | unit/sign from | free parameters |
|---|---|---|---|
| pilot_ridge (standard supervised readout) | ridge on pilot train, lambda by leave-one-writer-out over {.1,1,10,100,1000} | pilot classical calibration | 64 + 2 |
| pilot_pcr (source-learned subspace + anchor) | source SVD top-k, k by LOWO over {2,4,8,16,32}, regression on pilot | pilot | k + 2 |
| keyed | public lexicon sum | pilot | 2 |
| svd1 | top source principal component | pilot | 2 |
| stability_author | top Fisher author discriminant on source | pilot | 2 |
| stability_response | top random-slope component (between-author slope covariance vs sampling) | pilot | 2 |
| oracle_map (privileged) | least squares of TRUE source expression on x | true expression | 65 |

Classical calibration regresses the projection on the exact declared label (projection = alpha +
gamma label) so the evidence unit is unbiased for the declared expression. Without a label (W3) the
unit and sign are unidentified; label-free arms use the largest-|loading|-positive rule and report
correlations only.

Response kernel (standard, target-matched, shared by every arm): F04 noise-corrected
random-coefficient moments on source evidence (intercept + slope in z; spectral floor 1e-6) and the
Gaussian BLUP; the unshrunk per-author OLS coefficients are reported alongside. Targets: level =
intercept (conditions symmetric), Delta = 2 slope, held-condition mean m(1/3) = a + b/3 scored from
the other three conditions (W2-W4).

## Frozen source/pilot rules (computed without application truth)

1. Anchor support: |t| of the pilot calibration slope >= 4, else ANCHOR_UNSUPPORTED_DIRECTION.
2. Pilot design check: R^2 of label on pilot-writer dummies (train) > .5 flags CONFOUNDED_PILOT_DESIGN.
3. Unit tolerance tau_T = max(.15, 2 SE) where SE is the bootstrap SE (500 text draws) of the
   validation-writer slope of evidence on declared label.
4. Fingerprint tolerance tau_F = .20 (declared engineering choice, not a psychological threshold).
5. Nonconstancy: SD of application target scores >= .1 sqrt(t' Sigma_hat t) from the source fit.
6. Pilot uncertainty: 30 pilot-train-writer bootstrap refits (grid choices and label-free
   directions fixed, unit re-estimated); interval variance = mean posterior variance + between
   variance. 95% normal intervals.

## Evaluation (evaluator stage, after the estimator stage has saved scores and tolerances)

Cold application authors. Recovery vs hidden target: Pearson r (500 author-bootstrap CI), RMSE,
bias, calibration slope, interval coverage and width; paired r and RMSE differences vs pilot_ridge.
Interventions with common random numbers, rescored by the frozen estimator:
- target: a_u + 1 (level), b_u + .5 (Delta + 1);
- fingerprint: f_u + 2 along a random unit direction per author; style: r_u + 1;
- wording: new noise draws; material: topics from the held-out family {8..11}.
gain_T = mean change of the OLS target / intended change. R_F = RMS change of the OLS target under
a fingerprint intervention / |mean change under the target intervention|.
Analytic expectation stated in advance: a condition contrast cancels any additive author constant,
and the symmetric-condition level cancels the zero-mean style term, so R_F(level fingerprint) = 0
for Delta and R_F(style) = 0 for level; the informative checks are fingerprint for level and style
for Delta.

Pass per arm, world and seed: status OK, no confounded-design flag, nonconstant, |gain_T - 1| <=
tau_T, gain CI lower bound > 0, both R_F <= tau_F. An arm passes a world when it passes 3/3 seeds.
**Positive acceptance**: at least one permissible non-oracle arm passes W1 (level) and W2 (Delta).
W3 and W4 are reported as boundary controls. A failed target is not relabelled.

## Alternate checks

Vectorized kernel vs F04 `fit_source_moments` / `score_response`; analytic vs simulated
intervention effects; REML (`fit_source_lmm`) vs moments scores for one arm; isolation (estimator
outputs unchanged when evaluator truth is replaced).

## Disclosed pre-freeze scratch check

Seed 999 (not an evaluation seed) was run once to confirm the code executes and the designed
threats exist (stability directions load on fingerprint / style). It led to two changes before
freezing: the anchor-support rule (1) and classical unit calibration for the oracle.
