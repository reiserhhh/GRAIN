# Function-Aware Semantic Reader V1

2026-09-13. Frozen before this tranche's model fits/results. All-plan C05/C08,
C12/C13/C14/C16, following RESPONSE_ACCOUNTING_REPORT. A bounded construction
experiment on model affinities, not external personality validation.

## Question and Decision

Pointwise squared error does not determine the error of a prescribed profile
or contrast. Test whether matching the target functional during fitting helps
its held-author readout. Keep every comparison, including tradeoffs; do not
declare the highest held correlation a new universal scorer.

## Fixed Material

Reuse C14 semantic_link_r2 and response_accounting_r1: 624 authors, 36 original
comments each, 29,075 candidate sentences, 60 mean/max IPIP affinity targets.
No questionnaires, MEPS, bridge, fresh neural inference, downloads or uploads.
Verify input hashes, sentence/comment addresses, folds and condition operators.
Fold0 is historically explored development material, not a fresh lockbox.

Each comment averages its supported sentence features AFTER any polynomial
expansion; teacher targets are averaged on exactly the same sentences. Missing
comments stay missing. Authors weight equally, supported comments equally
within author. Teacher centering/SD and all coordinate scalers/category support
are fitted only on source sentences, using equal total author weights.

## Models and Source-Only Selection

Two coordinate maps: joint_linear and joint_quadratic, same 24 external frozen
GRAIN lexical units and opportunity features as C14. They are unit adapters,
not the whole GRAIN system. No new feature search.

Define e_u = X_u B - T_u on supported comment rows, with an explicit unpenalized
intercept column in X. D_u gives each supported comment weight 1/n_u.
A_level averages all supported comments; A_total and A_within are the previous
chronological operators, with strict time boundaries and shared support for W.

L_point = mean_u tr(e_u' D_u e_u)/Q.
L_j = mean_eligible_u ||A_uj e_u||^2 / (Q s_j^2), where
s_j^2 = max(.01, mean_eligible_source_u,q (A_uj T_uq)^2), in source teacher SD.
These source-frozen scalar energies normalize measurement units, not validity.

Five arms, no lambda grid:

- point: L_point.
- level: (L_point + L_level)/2.
- total: (L_point + L_total)/2.
- within: (L_point + L_within)/2.
- balanced: (L_point + L_level + L_total + L_within)/4.

Add alpha ||B_nonintercept||_F^2/Q. For each map, choose alpha from
[.0001,.001,.01,.1] using POINT validation loss alone: fit folds2/3/4, validate
fold1, deterministic lower-alpha tie order. Freeze that alpha for ALL five
arms of that map to isolate changed loss weighting. Refit all on folds1/2/3/4;
evaluate all frozen models on fold0. No held selection, retries or promotion.
Record validation tradeoffs for the five arms too, without choosing a winner.

Weighted stacked least squares gives (X' M X + alpha P) B = X' M T.
We do not interpret the design metric M as inverse random-error covariance.
These extra rows are deterministic functions of the same targets, not extra
independent observations. No sample-size gain or causal derivative is claimed.
Unlike smoothing/consistency penalties toward zero, A X B is matched to A T;
observed change is preserved rather than penalized away.

## Evaluation and Safeguards

Report author-balanced comment MSE/r; author level, total, within, composition,
turnover MSE/r for all 60 outputs and both affinity functions. The same frozen
source units and support masks apply to all arms. Paired 500-author bootstrap
of MSE improvement over the matching point arm, seed915, by estimand; intervals
are descriptive conditional-on-fit, not multiplicity-adjusted selection tests.

Assess random-partition total/within error (ten existing seeds), reporting
within-author averages before any cohort summaries. This checks dependence
on the training functional; it is not a new independent measurement occasion.
One source-author target-panel permutation for balanced per coordinate map,
fixed seed915. Transfer complete 36-position target panels; intersect receiver
and donor observed comments and rebuild operators, never insert missing as0.
This control includes residual common-format/time structure; report its actual
support and scores, not a formal null p-value.

Audit stacked least squares against normal equations and stationarity, source
isolation, missing/shared support, intercept contrast cancellation, reload and
prediction aggregation, plus a synthetic known-answer functional tradeoff.
Use a direct arithmetic audit of the real outputs and holdout hashes.

## Delivery / Stop

Ignored output all_plan_closure_v1/functional_reader_r1: contract, tuning,
frozen models/selection, private inputs/predictions, all metrics, paired
contrasts, random controls, source supports/energies, audit, summary, errors.
Public report describes the construction implication and the remaining
ownership/semantic-validity gap. Local CPU, two BLAS threads, 20 minutes max.
Stop after the declared comparison; no higher-degree or larger-target grid.
Keep the all-plan goal active and move to the next distinct unresolved question.

## Sources

[Czarnecki et al. 2017, Sobolev Training](https://proceedings.neurips.cc/paper_files/paper/2017/file/758a06618c69880a6cee5314ee42d52f-Paper.pdf):
read Introduction, section2 eq1-2 and theory motivation. They match derivatives;
we match finite support-dependent linear averages/contrasts. Their derivative
sample-complexity arguments do not transfer to deterministic reused rows here.
[GNU GSL weighted/Tikhonov least squares](https://www.gnu.org/software/gsl/doc/html/lls.html):
normal-equation identity checked from official indexed documentation; full-page
open failed. Our solver is ordinary augmented weighted least squares.
