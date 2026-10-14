# F16-R03-TEXT: actual reader replacement, frozen design

Date: 2026-09-24. This extends, not replaces, the original F12/F16 requirements.
No application results have been read for this design. No new human data,
questionnaires, API calls, neural training, model selection or prompt tuning.

## Intervention and material support

Replace the qualification reader, keeping content and texts fixed:
old = pinned NLI polarity + frozen historical GRAIN comment-view qualification;
new = the same NLI polarity + its already-frozen actual-self NLI hypothesis.
These are actual text computations, not transformations of old numeric scores.
Neither model nor hypothesis is fitted to the new bank.

Create 380 distinct public controlled English materials before inference:
80 source reference, 100 paired pilot (first60 calibration, last40 validation),
200 application. Role-specific wrappers are disjoint; semantic action/status
templates remain shared. This is a finite controlled-material test, not independent
natural-language or human validation. The pilot unit is a material, NOT a person;
the original numeric 100-person pilot is not relabeled as text evidence. Every
role and pilot split covers all four (pole, qualification) strata. Quoted and
hypothetical statements are unqualified; actual and corrected statements qualify.
Raw text hashes are disjoint from the old96 bank and between new roles.
Read all380 with the existing three NLI hypotheses (1140 local pairs) and frozen
GRAIN transform/head. Preserve raw probabilities and every input, no rescue loop.

## Observation law and arms

Observed category c=2*I(pole>=.5)+I(qualification>=.5). Calibrate E[b,q,c].
Qualified pole follows the unchanged probit observation model; unqualified pole
is an independent fair bit and eligibility has design probability .5. The latent
unit is inherited from the frozen numeric source model, not identified by text.
Rows use Jeffreys Dirichlet(count+.5); plugin is its mean. Validation is diagnostic
only. Missing a true pole/qualification stratum returns MISSING_ANCHOR, not an
invented uniform likelihood. Condition-specific claims are not made by a pooled
channel. Pilot empirical rank and conditional content contrasts are reported.

Six fixed arms on the SAME newly rendered histories:
1. old_matched: old reader + old source-material channel mixture.
2. changed_uncalibrated: new reader + that old channel mixture.
3. changed_pilot_plugin: new reader + pilot mean channel.
4. changed_pilot_mixture: new reader + pilot channel draws (primary).
5. changed_known_pool: exact category frequencies in the finite application pool;
   labeled information reference, forbidden as an operational calibrator.
6. oracle_labels: true rendered pole/qualification, labeled information reference.

Use all200 frozen source components and 200 independently generated channel draws,
paired one-per-source as a cut-mixture Monte Carlo estimate of the independent
source x channel law (NOT all40000 combinations). Source/reader spread is combined;
do not call their difference from plugin a pure variance component. Alternate
channel draws and doubled quadrature on first/last application IDs diagnose
Monte Carlo error without tuning or selecting primary scores. This packet retains
the old categorical QMC/ADF estimators and their approximation limits.

Three original seeds, 200 application authors each for T1 and dynamic T2/T3.
T2/T3 share authors; do not add them as distinct people. Reuse actual F04-L and
F05-core histories/source fits, 32 and48 observations respectively. Render with a
new role-separated RNG; preserve all conditions, empty clocks and future plan
(+1,-1,+1,-1). No application latent values enter scoring. Truth is read only by
the separate final evaluation command. This reuses SIM authors, not fresh human
replication. Raw target units and unchanged historical reference provenance are
kept; no new application-batch standardization.

## Outputs and interpretation

Freeze code/input hashes and bank before reading; save actual raw reader outputs,
source/pilot/application roles, counts/channels, validation log loss, missing-anchor
controls, all six predictions, target-wise MSE/bias/r/coverage/width and paired
author-bootstrap delta MSE (2000 draws stratified by seed). Intervals condition on
this shared finite material bank and fitted artifacts; materials are not hundreds
of independent human utterances. Also show whether empirical channel rank drops,
whether the content contrast survives, and which designed statuses cause errors.
Known-pool success with pilot failure suggests calibration/transport limits;
known-pool and true-label separation quantifies reader loss at this design.
No pass requires all arms to improve. Retain failures and report implementation,
supported mechanism, counterexample and pending public serialization separately.
No source-response refit or new norm fit in this packet: raw target scores
are compared; historical standardized reference positions are not interpreted.

Verify counts by an independent crosstab, first/last scores by alternate law
enumeration, mixture moments by raw second moments, material/address isolation,
application metrics and source/artifact hashes. A fresh process replays fixed
inputs. Full public calibration-bundle serialization remains the next packet.
